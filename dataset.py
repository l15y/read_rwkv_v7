import math
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset
from .utils import MaybeIsPrime

# 自定义数据集类，用于处理大规模训练数据
class MyDataset(Dataset):
    def __init__(self, args):
        # 初始化数据集
        self.args = args
        self.vocab_size = args.vocab_size
        rank_zero_info(f"当前词汇表大小 = {self.vocab_size} (请确保正确)")

        # 根据pile版本加载数据
        if args.my_pile_version == 1:
            # 版本1：使用MMapIndexedDataset加载数据
            self.data = MMapIndexedDataset(args.data_file)
            self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
            rank_zero_info(f"数据包含 {self.data_size} 个token")
        elif args.my_pile_version == 2:
            # 版本2：从文本文件加载数据
            data_list = open(args.data_file, "r", encoding='utf-8').read().strip().split('\n')
            data_list = [i.strip().split(' ') for i in data_list]
            self.data = []
            self.data_size = int(data_list[-1][-1])
            rank_zero_info(f"数据包含 {self.data_size} 个chunk")
            for d in data_list:
                data = MMapIndexedDataset(d[0])
                data_size = len(data._bin_buffer) // data._index._dtype_size
                assert (data_size - args.ctx_len) == int(d[1])
                self.data += [[int(d[-1]), int(d[1]), data]]
        
        # 初始化pile相关变量
        self.data_pile = None
        self.data_pile_size = 0

        # 处理pile stage相关逻辑
        if args.my_pile_stage > 0:
            self.samples_per_epoch = args.epoch_steps * args.real_bsz
            assert self.samples_per_epoch == 40320
            rank_zero_info(f"########## Pile 20b-tokenized stage {args.my_pile_stage} ##########")
            dataset_slot = self.data_size // args.ctx_len
            if args.my_pile_stage != 4:
                # 验证magic prime参数
                assert MaybeIsPrime(args.magic_prime)
                assert args.magic_prime % 3 == 2
                assert args.magic_prime / dataset_slot > 0.9 and args.magic_prime / dataset_slot <= 1

    def __len__(self):
        # 返回数据集长度
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        # 获取单个训练样本
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        ctx_len = args.ctx_len
        req_len = ctx_len + 1
        magic_prime = args.magic_prime
        data = self.data

        # 处理pile stage采样逻辑
        if args.my_pile_stage > 0:
            ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank
            if data == self.data_pile:
                # 从pile数据中随机采样
                i = np.random.randint(0, self.data_pile_size - req_len)
            else:
                if args.my_pile_stage == 4 or ii < args.my_random_steps:
                    # 随机采样模式
                    if args.my_pile_version == 1:
                        i = np.random.randint(0, self.data_size - req_len)
                    else:
                        i = np.random.randint(0, self.data_size)
                else:
                    # 使用magic prime进行确定性采样
                    ii = ii - args.my_random_steps
                    factor = (math.sqrt(5) - 1) / 2
                    factor = int(magic_prime * factor)
                    i = ((factor * ii * ii * ii) % magic_prime) * ctx_len
                    i = i + args.my_pile_shift
        else:
            # 普通随机采样
            i = np.random.randint(0, self.data_size - req_len)
        
        # 获取实际数据
        if args.my_pile_version == 1:
            dix = data.get(idx=0, offset=i, length=req_len).astype(int)
        else:
            # 处理版本2的数据结构
            for j in range(len(data)):
                if i < data[j][0]:
                    ii = i
                    i = (i - (data[j-1][0] if j > 0 else 0)) % data[j][1]
                    dix = data[j][2].get(idx=0, offset=i, length=req_len).astype(int)
                    break
        
        # 返回输入和标签
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
