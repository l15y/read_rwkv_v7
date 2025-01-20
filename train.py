import logging
logging.basicConfig(level=logging.INFO)

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
import pytorch_lightning as pl

# 参数解析器
parser = ArgumentParser()

# 添加训练参数
parser.add_argument("--load_model", default="", type=str)  # 模型加载路径，包含.pth扩展名
parser.add_argument("--wandb", default="", type=str)  # wandb项目名称，如果为空则不使用wandb
parser.add_argument("--proj_dir", default="out", type=str)  # 项目输出目录
parser.add_argument("--random_seed", default="-1", type=int)  # 随机种子
parser.add_argument("--train_type", default="", type=str) # 训练类型 ""/"states"

# 数据相关参数
parser.add_argument("--data_file", default="", type=str)  # 数据文件路径
parser.add_argument("--vocab_size", default=0, type=int)  # 词汇表大小，0表示自动检测

# 模型结构参数
parser.add_argument("--ctx_len", default=1024, type=int)  # 上下文长度
parser.add_argument("--epoch_steps", default=1000, type=int)  # 每个mini epoch的步数
parser.add_argument("--epoch_count", default=500, type=int)  # 训练的总epoch数
parser.add_argument("--epoch_begin", default=0, type=int)  # 起始epoch数
parser.add_argument("--epoch_save", default=5, type=int)  # 保存模型的间隔epoch数

parser.add_argument("--micro_bsz", default=12, type=int)  # 每个GPU的微批次大小
parser.add_argument("--n_layer", default=6, type=int)  # 模型层数
parser.add_argument("--n_embd", default=512, type=int)  # 嵌入维度
parser.add_argument("--dim_att", default=0, type=int)  # 注意力维度
parser.add_argument("--dim_ffn", default=0, type=int)  # FFN维度
parser.add_argument("--head_qk", default=0, type=int)  # headQK技巧
parser.add_argument("--tiny_att_dim", default=0, type=int)  # 小型注意力维度
parser.add_argument("--tiny_att_layer", default=-999, type=int)  # 小型注意力所在层

# 优化器参数
parser.add_argument("--lr_init", default=6e-4, type=float)  # 初始学习率
parser.add_argument("--lr_final", default=1e-5, type=float)  # 最终学习率
parser.add_argument("--warmup_steps", default=-1, type=int)  # warmup步数
parser.add_argument("--beta1", default=0.9, type=float)  # Adam beta1
parser.add_argument("--beta2", default=0.99, type=float)  # Adam beta2
parser.add_argument("--adam_eps", default=1e-18, type=float)  # Adam epsilon
parser.add_argument("--grad_cp", default=0, type=int)  # 梯度检查点
parser.add_argument("--dropout", default=0, type=float)  # dropout率
parser.add_argument("--weight_decay", default=0, type=float)  # 权重衰减
parser.add_argument("--weight_decay_final", default=-1, type=float)  # 最终权重衰减
parser.add_argument("--grad_clip", default=1.0, type=float)  # 梯度裁剪

# 特殊训练模式参数
parser.add_argument("--my_pile_version", default=1, type=int)  # 特殊pile版本
parser.add_argument("--my_pile_stage", default=0, type=int)  # 特殊pile阶段
parser.add_argument("--my_pile_shift", default=-1, type=int)  # 文本shift
parser.add_argument("--my_pile_edecay", default=0, type=int)  # epoch衰减
parser.add_argument("--layerwise_lr", default=1, type=int)  # 分层学习率
parser.add_argument("--ds_bucket_mb", default=200, type=int)  # deepspeed bucket大小

# 其他参数
parser.add_argument("--my_sample_len", default=0, type=int)
parser.add_argument("--my_ffn_shift", default=1, type=int)
parser.add_argument("--my_att_shift", default=1, type=int)
parser.add_argument("--head_size_a", default=64, type=int)  # 头大小
parser.add_argument("--head_size_divisor", default=8, type=int)
parser.add_argument("--load_partial", default=0, type=int)
parser.add_argument("--magic_prime", default=0, type=int)
parser.add_argument("--my_random_steps", default=0, type=int)
parser.add_argument("--my_exit", default=99999999, type=int)
parser.add_argument("--my_exit_tokens", default=0, type=int)

# 添加Trainer参数
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

########################################################################################################

import os, warnings, math, datetime, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader
if "deepspeed" in args.strategy:
    import deepspeed
from pytorch_lightning import seed_everything

# 设置随机种子
if args.random_seed >= 0:
    print(f"########## 警告: 全局种子 {args.random_seed} 这将影响多GPU采样 ##########\n" * 3)
    seed_everything(args.random_seed)

# 设置numpy打印选项
np.set_printoptions(precision=4, suppress=True, linewidth=200)
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")

# 初始化训练参数
args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
args.enable_checkpointing = False
args.replace_sampler_ddp = False
args.logger = False
args.gradient_clip_val = args.grad_clip
args.num_sanity_val_steps = 0
args.check_val_every_n_epoch = int(1e20)
args.log_every_n_steps = int(1e20)
args.max_epochs = -1  # 无限训练
args.betas = (args.beta1, args.beta2)
args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
os.environ["RWKV_TRAIN_TYPE"] = args.train_type
if args.dim_att <= 0:
    args.dim_att = args.n_embd
args.dim_ffn = int((args.n_embd * 4) // 32 * 32)
args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
if not os.path.exists(args.proj_dir):
    os.makedirs(args.proj_dir)

# 特殊pile阶段处理
if args.my_pile_stage > 0:
    magic_prime_bak = args.magic_prime
    if args.my_pile_shift < 0:
        args.my_pile_shift = 0
    if magic_prime_bak > 0:
        args.magic_prime = magic_prime_bak
    args.epoch_count = args.magic_prime // 40320
    args.epoch_steps = 40320 // args.real_bsz
    assert args.epoch_steps * args.real_bsz == 40320

    # 查找最新保存的模型
    if args.my_pile_stage >= 2 and len(args.load_model) == 0:
        list_p = []
        for p in os.listdir(args.proj_dir):
            if p.startswith("rwkv") and p.endswith(".pth"):
                p = ((p.split("-"))[1].split("."))[0]
                if p != "final":
                    if p == "init":
                        p = -1
                    else:
                        p = int(p)
                    list_p += [p]
        list_p.sort()
        max_p = list_p[-1]
        if len(list_p) > 1:
            args.my_pile_prev_p = list_p[-2]  # 如果max_p损坏则使用前一个
        if max_p == -1:
            args.load_model = f"{args.proj_dir}/rwkv-init.pth"
        else:
            args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
            if args.warmup_steps < 0:
                if args.my_pile_stage == 2:
                    args.warmup_steps = 10
                else:
                    args.warmup_steps = 30
        args.epoch_begin = max_p + 1
    else:
        args.epoch_begin = 0

# 计算每个epoch的样本和token数
samples_per_epoch = args.epoch_steps * args.real_bsz
tokens_per_epoch = samples_per_epoch * args.ctx_len
try:
    deepspeed_version = deepspeed.__version__
except:
    deepspeed_version = None
    pass

# 打印训练信息
rank_zero_info(
    f"""
############################################################################
#
# RWKV-7 {args.precision.upper()} on {args.num_nodes}x{args.devices} {args.accelerator.upper()}, bsz {args.num_nodes}x{args.devices}x{args.micro_bsz}={args.real_bsz}, {args.strategy} {'with grad_cp' if args.grad_cp > 0 else ''}
#
# Data = {args.data_file} (binidx), ProjDir = {args.proj_dir}
#
# Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (will continue afterwards), save every {args.epoch_save} epoch
#
# Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
#
# Model = {args.n_layer} n_layer, {args.n_embd} n_embd, {args.ctx_len} ctx_len
#
# Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
#
# Found torch {torch.__version__}, recommend latest torch
# Found deepspeed {deepspeed_version}, recommend latest deepspeed
# Found pytorch_lightning {pl.__version__}, recommend 1.9.5
#
############################################################################
"""
)
rank_zero_info(str(vars(args)) + "\n")

# 学习率调度提示
if args.lr_final == 0 or args.lr_init == 0:
    rank_zero_info("\n\n注意: lr_final = 0 或 lr_init = 0。将使用线性学习率调度。\n\n")

# 精度设置
assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
os.environ["RWKV_FLOAT_MODE"] = args.precision
if args.precision == "fp32":
    for i in range(10):
        rank_zero_info("\n\n注意: 你正在使用fp32（非常慢）。尝试使用bf16/tf32以获得更快的训练。\n\n")
if args.precision == "fp16":
    rank_zero_info("\n\n注意: 你正在使用fp16（可能会溢出）。尝试使用bf16/tf32以获得更稳定的训练。\n\n")

# JIT设置
os.environ["RWKV_JIT_ON"] = "1"
if "deepspeed_stage_3" in args.strategy:
    os.environ["RWKV_JIT_ON"] = "0"

# CUDA设置
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
if args.precision == "fp32":
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
else:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

# 精度转换
if "32" in args.precision:
    args.precision = 32
elif args.precision == "fp16":
    args.precision = 16
else:
    args.precision = "bf16"

########################################################################################################

from trainer import train_callback, generate_init_weight
from dataset import MyDataset

# 初始化数据集
train_data = MyDataset(args)
args.vocab_size = train_data.vocab_size

# 初始化模型
from model import RWKV
model = RWKV(args)

# 生成初始权重
if len(args.load_model) == 0 or args.my_pile_stage == 1:
    init_weight_name = f"{args.proj_dir}/rwkv-init.pth"
    generate_init_weight(model, init_weight_name)
    args.load_model = init_weight_name

# 加载模型
rank_zero_info(f"########## 正在加载 {args.load_model}... ##########")
try:
    load_dict = torch.load(args.load_model, map_location="cpu")
    load_keys = list(load_dict.keys())
except:
    rank_zero_info(f"错误的检查点 {args.load_model}")
    if args.my_pile_stage >= 2:  # 尝试使用另一个检查点
        max_p = args.my_pile_prev_p
        if max_p == -1:
            args.load_model = f"{args.proj_dir}/rwkv-init.pth"
        else:
            args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
        args.epoch_begin = max_p + 1
        rank_zero_info(f"尝试 {args.load_model}")
        load_dict = torch.load(args.load_model, map_location="cpu")

# 部分加载模型
if args.load_partial == 1:
    load_keys = load_dict.keys()
    for k in model.state_dict():
        if k not in load_keys:
            load_dict[k] = model.state_dict()[k]
model.load_state_dict(load_dict, strict=False)

# 初始化Trainer
trainer = Trainer.from_argparse_args(
    args,
    callbacks=[train_callback(args)],
)

# 打印模型参数
if trainer.global_rank == 0:
    for n in model.state_dict():
        shape = model.state_dict()[n].shape
        s0 = str(shape[0]) if len(shape) > 0 else ""
        s1 = str(shape[1]) if len(shape) > 1 else ""
        s2 = str(shape[2]) if len(shape) > 2 else ""
        s3 = str(shape[3]) if len(shape) > 3 else ""
        print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}")

# DeepSpeed配置
if "deepspeed" in args.strategy:
    trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
    trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = args.ds_bucket_mb * 1000 * 1000

# 初始化数据加载器
data_loader = DataLoader(train_data, shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)

# 开始训练
trainer.fit(model, data_loader)
