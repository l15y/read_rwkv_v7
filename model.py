########################################################################################################
# https://github.com/l15y/read_rwkv_v7/tree/main
# RWKV 语言模型 - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib  # 导入基础库
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam


def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


########################################################################################################
# CUDA 内核
# 这部分代码负责加载和初始化CUDA内核，用于加速模型计算
########################################################################################################

from torch.utils.cpp_extension import load  # 用于加载自定义CUDA扩展

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])  # 从环境变量获取头大小

CHUNK_LEN = 16  # 设置块长度

# 编译标志，用于优化CUDA内核性能
flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
# 加载自定义CUDA内核
load(name="wind_backstepping", sources=[f'cuda/wkv7_cuda.cu', 'cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

class WindBackstepping(torch.autograd.Function):
    """自定义CUDA内核的封装类
    
    实现了RWKV特有的时间注意力机制的高效计算。
    通过CUDA内核加速，实现了线性复杂度的时间注意力计算。
    """
    @staticmethod
    def forward(ctx, w,q,k,v,z,b):
        """前向传播函数
        
        参数:
            ctx: 上下文对象，用于保存反向传播所需信息
            w: 权重矩阵，形状为 (batch_size, seq_len, head_size, head_size)
            q: 查询矩阵，形状为 (batch_size, seq_len, head_size, head_size)
            k: 键矩阵，形状为 (batch_size, seq_len, head_size, head_size)
            v: 值矩阵，形状为 (batch_size, seq_len, head_size, head_size)
            z: 辅助矩阵1，形状为 (batch_size, seq_len, head_size, head_size)
            b: 辅助矩阵2，形状为 (batch_size, seq_len, head_size, head_size)
            
        返回:
            torch.Tensor: 计算结果，形状与v相同
        """
        B,T,H,C = w.shape  # 获取输入张量的形状：批大小、序列长度、头数、头大小
        assert T%CHUNK_LEN == 0  # 确保序列长度能被块长度整除
        assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])  # 检查所有输入张量的数据类型
        assert all(i.is_contiguous() for i in [w,q,k,v,z,b])  # 检查所有输入张量是否连续存储
        y = torch.empty_like(v)  # 初始化输出张量
        s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)  # 初始化状态张量
        sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)  # 初始化辅助状态张量
        torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)  # 调用CUDA内核进行计算
        ctx.save_for_backward(w,q,k,v,z,b,s,sa)
        return y
    @staticmethod
    def backward(ctx, dy):
        """反向传播函数
        
        参数:
            ctx: 上下文对象，包含前向传播保存的信息
            dy: 输出梯度，形状与前向传播的输出相同
            
        返回:
            tuple: 包含各输入参数的梯度
        """
        assert all(i.dtype==torch.bfloat16 for i in [dy])  # 检查梯度张量的数据类型
        assert all(i.is_contiguous() for i in [dy])  # 检查梯度张量是否连续存储
        w,q,k,v,z,b,s,sa = ctx.saved_tensors  # 获取前向传播保存的张量
        dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]  # 初始化各参数的梯度张量
        torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)  # 调用CUDA内核进行反向计算
        return dw,dq,dk,dv,dz,db

def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
    """调用CUDA内核进行RWKV时间注意力计算
    
    参数：
        q: 查询矩阵，形状为 (batch_size, seq_len, hidden_size)
        w: 权重矩阵，形状为 (batch_size, seq_len, hidden_size)
        k: 键矩阵，形状为 (batch_size, seq_len, hidden_size)
        v: 值矩阵，形状为 (batch_size, seq_len, hidden_size)
        a: 辅助矩阵1，形状为 (batch_size, seq_len, hidden_size)
        b: 辅助矩阵2，形状为 (batch_size, seq_len, hidden_size)
        
    返回：
        torch.Tensor: 计算结果，形状为 (batch_size, seq_len, hidden_size)
        
    实现细节：
    1. 将输入张量reshape为适合CUDA内核的形状
    2. 调用WindBackstepping.apply进行计算
    3. 将结果reshape回原始形状
    """
    B,T,HC = q.shape
    q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
    return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)


class RWKV_Tmix_x070(MyModule):
    """RWKV 时间混合模块 x070 版本
    这是模型的核心组件之一，负责处理时间序列数据的混合和特征提取。
    该模块实现了RWKV架构特有的时间混合机制，结合了RNN和Transformer的优点。
    
    主要功能：
    - 通过时间偏移机制捕捉序列中的时间依赖关系
    - 使用可学习的参数控制不同特征（接收、键、值等）的混合比例
    - 实现了一个高效的时间注意力机制，避免了传统Transformer的平方复杂度
    
    参数：
        args: 模型配置参数对象
        layer_id: 当前层在模型中的索引
        
    属性：
        head_size: 每个注意力头的大小
        n_head: 注意力头的数量
        x_r, x_w, x_k, x_v, x_a, x_g: 可学习的时间混合参数
        w0, w1, w2: 权重计算相关参数
        a0, a1, a2: 注意力缩放相关参数
        v0, v1, v2: 值残差连接相关参数
        g1, g2: 门控机制相关参数
        time_shift: 时间偏移层
        receptance, key, value, output: 线性变换层
        ln_x: 层归一化
    """
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args  # 模型参数
        self.layer_id = layer_id  # 当前层ID
        self.my_testing = args.my_testing  # 测试模式标志

        # 初始化注意力头相关参数
        self.head_size = args.head_size_a  # 每个注意力头的大小
        self.n_head = args.dim_att // self.head_size  # 注意力头数量
        assert args.dim_att % self.n_head == 0  # 确保维度可被头数整除
        H = self.n_head  # 头数
        N = self.head_size  # 头大小
        C = args.n_embd  # 嵌入维度

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1  # 计算当前层在总层数中的比例
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            # D_DECAY_LORA = 64
            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)  # 初始化衰减速度张量
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)  # 计算每个维度的衰减速度
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            # D_AAA_LORA = 64
            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            # D_MV_LORA = 32
            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            # D_GATE_LORA = 128
            D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(args.head_size_divisor**2)) # !!! notice eps value !!!


    @MyFunction
    def forward(self, x, v_first):
        """前向传播函数
        
        参数:
            x: 输入张量，形状为 (batch_size, seq_len, hidden_size)
                包含当前时间步的输入特征
            v_first: 第一层的值向量
                用于残差连接的特殊值向量，仅在layer_id=0时使用
                
        返回:
            tuple: 包含两个元素
                - 输出张量，形状与输入x相同
                - 更新后的值向量，用于下一层或下一次前向传播
                
        实现细节：
        1. 计算时间偏移特征，捕捉序列中的时间依赖关系
        2. 使用可学习参数混合不同特征（接收、键、值等）
        3. 计算注意力权重，应用soft-clamp限制范围
        4. 处理第一层的特殊情况，更新值向量
        5. 调用CUDA内核进行高效计算
        6. 应用层归一化和残差连接
        7. 使用门控机制控制信息流动
        """
        B, T, C = x.size()  # 获取输入形状：批大小、序列长度、特征维度
        H = self.n_head  # 注意力头数
        
        # 时间偏移计算，用于捕捉时间序列中的模式
        xx = self.time_shift(x) - x

        # 计算不同特征变换
        xr = x + xx * self.x_r  # 接收特征
        xw = x + xx * self.x_w  # 权重特征
        xk = x + xx * self.x_k  # 键特征
        xv = x + xx * self.x_v  # 值特征
        xa = x + xx * self.x_a  # 注意力特征
        xg = x + xx * self.x_g  # 门控特征

        # 计算各个组件
        r = self.receptance(xr)  # 接收门
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5  # 权重计算，使用soft-clamp限制范围
        k = self.key(xk)  # 键
        v = self.value(xv)  # 值
        
        # 处理第一层特殊情况
        if self.layer_id == 0:
            v_first = v  # 存储第一层的值向量
        else:
            # 添加值残差连接
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)  # 计算上下文学习率
        g = torch.sigmoid(xg @ self.g1) @ self.g2  # 计算门控值

        # 键处理
        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)  # 归一化
        k = k * (1 + (a-1) * self.k_a)  # 应用注意力缩放

        # 调用CUDA内核进行计算
        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a)
        # 层归一化
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        # 残差连接和输出
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)  # 应用门控并输出
        return x, v_first
    
class RWKV_CMix_x070(MyModule):
    """RWKV 通道混合模块 x070 版本
    
    实现通道间的特征混合，通过非线性变换增强特征表达能力。
    主要特点：
    - 使用ReLU激活函数和平方操作增强非线性
    - 通过线性变换实现特征混合
    - 支持不同层使用不同的混合参数
    """
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)


    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)


class Block(nn.Module):
    """RWKV模型的基本构建块
    
    每个Block包含：
    - 层归一化
    - 时间混合模块（RWKV_Tmix_x070）
    - 通道混合模块（RWKV_CMix_x070）
    - 可选的小注意力机制
    
    参数：
        args: 模型配置参数
        layer_id: 当前层在模型中的索引
        
    属性：
        ln1, ln2: 层归一化模块
        ln0: 仅在第一层使用的额外层归一化
        pos_emb_x, pos_emb_y: 位置编码（如果启用）
        att: 时间混合模块
        ffn: 通道混合模块
        tiny_ln, tiny_q, tiny_k, tiny_v: 小注意力机制相关组件（如果启用）
        drop0, drop1: dropout层（如果启用）
    """
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)
            if args.my_pos_emb > 0:
                self.pos_emb_x = nn.Parameter(torch.zeros((1,args.my_pos_emb,args.n_embd)))
                self.pos_emb_y = nn.Parameter(torch.zeros((args.my_pos_emb,1,args.n_embd)))

        if self.layer_id == 0 and self.args.pre_ffn > 0:
            self.ffnPre = RWKV_ChannelMix(args, 0)
        else:
            self.att = RWKV_Tmix_x070(args, layer_id)

        self.ffn = RWKV_CMix_x070(args, layer_id)
        
        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            self.tiny_ln = nn.LayerNorm(args.n_embd)
            self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.register_buffer("tiny_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)

    def forward(self, x, v_first):
        """Block前向传播
        
        参数：
            x: 输入特征，形状为 (batch_size, seq_len, hidden_size)
            v_first: 第一层的值向量
            
        返回：
            tuple: 包含两个元素
                - 输出特征，形状与输入相同
                - 更新后的值向量
                
        计算流程：
        1. 如果是第一层，应用额外的层归一化
        2. 通过时间混合模块处理特征
            - 先对输入进行层归一化
            - 应用时间混合
            - 添加残差连接
        3. 通过通道混合模块处理特征
            - 先对输入进行层归一化
            - 应用通道混合
            - 添加残差连接
        4. 返回处理后的特征和更新后的值向量
        """
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn

        x = x + self.ffn(self.ln2(x))
        return x, v_first

class L2Wrap(torch.autograd.Function):
    """L2正则化封装类
    
    实现自定义的L2正则化计算，用于鼓励logits接近0。
    主要功能：
    - 在前向传播中直接返回损失
    - 在反向传播中计算梯度并应用L2正则化
    """
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV(pl.LightningModule):
    """RWKV 主模型类
    继承自PyTorch Lightning的Module，包含完整的模型架构。
    实现了基于RWKV架构的语言模型，结合了RNN和Transformer的优点。
    
    主要特点：
    - 线性复杂度的时间注意力机制
    - 可扩展的深度架构
    - 支持多种浮点精度模式（fp32, fp16, bf16）
    - 集成PyTorch Lightning的训练框架
    
    模型结构：
    1. 输入嵌入层
    2. 多层RWKV模块堆叠
    3. 输出层归一化
    4. 输出投影层
    
    参数：
        args: 模型配置参数对象，包含以下关键参数：
            - vocab_size: 词汇表大小
            - n_embd: 嵌入维度
            - n_layer: 层数
            - ctx_len: 上下文长度
            - dim_att: 注意力维度
            - dim_ffn: 前馈网络维度
            - head_qk: 头注意力维度
            - dropout: dropout概率
            - tiny_att_dim: 小注意力维度
            - tiny_att_layer: 使用小注意力的层索引
    """
    """RWKV 主模型类
    继承自PyTorch Lightning的Module，包含完整的模型架构。
    实现了基于RWKV架构的语言模型，结合了RNN和Transformer的优点。
    
    主要特点：
    - 线性复杂度的时间注意力机制
    - 可扩展的深度架构
    - 支持多种浮点精度模式（fp32, fp16, bf16）
    - 集成PyTorch Lightning的训练框架
    
    模型结构：
    1. 输入嵌入层
    2. 多层RWKV模块堆叠
    3. 输出层归一化
    4. 输出投影层
    
    参数：
        args: 模型配置参数对象，包含以下关键参数：
            - vocab_size: 词汇表大小
            - n_embd: 嵌入维度
            - n_layer: 层数
            - ctx_len: 上下文长度
            - dim_att: 注意力维度
            - dim_ffn: 前馈网络维度
            - head_qk: 头注意力维度
            - dropout: dropout概率
            - tiny_att_dim: 小注意力维度
            - tiny_att_layer: 使用小注意力的层索引
    """
    def __init__(self, args):
        super().__init__()
        self.args = args  # 模型参数
        
        # 初始化模型维度参数
        if not hasattr(args, 'dim_att'):
            args.dim_att = args.n_embd  # 默认注意力维度等于嵌入维度
        if not hasattr(args, 'dim_ffn'):
            if '-f4' in os.environ["RWKV_MY_TESTING"]:
                args.dim_ffn = int((args.n_embd * 4) // 32 * 32)  # 测试模式下使用4倍维度
            else:
                args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)  # 默认使用3.5倍维度
            
        # 初始化小注意力机制参数
        if not hasattr(args, 'tiny_att_layer'):
            args.tiny_att_layer = -1  # 默认禁用小注意力
        if not hasattr(args, 'tiny_att_dim'):
            args.tiny_att_dim = -1  # 默认小注意力维度
            
        # 确保维度是32的倍数，便于硬件优化
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        if args.head_qk > 0:
            self.head_q = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.head_k = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.register_buffer("copy_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
#LR和wd
    def configure_optimizers(self):
        args = self.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():

            if args.train_type == 'states':
                if 'time_sta' not in n:
                    continue

            if (("_w1" in n) or ("_w2" in n)) and (args.layerwise_lr > 0):
                lr_1x.add(n)
            elif (("time_sta" in n) and (args.weight_decay > 0)):
                lr_decay.add(n)
            elif (("time_mix" in n) or ("time_maa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif (("time_decay" in n) or ("time_daaaa" in n) or ("att.w0" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (args.layerwise_lr > 0):
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0) and (".weight" in n):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))

        if self.trainer.is_global_zero:
            print('decay', lr_decay, '\n')
            print('1x', lr_1x, '\n')
            print('2x', lr_2x, '\n')
            print('3x', lr_3x, '\n')

        param_dict = {n: p for n, p in self.named_parameters()}
        
        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(self, idx):
        """模型前向传播
        
        参数:
            idx: 输入token索引，形状为 (batch_size, seq_len)
                包含要处理的token序列的索引
                
        返回:
            torch.Tensor: 模型输出logits，形状为 (batch_size, seq_len, vocab_size)
                表示每个位置每个token的未归一化对数概率
                
        计算流程：
        1. 检查输入序列长度是否超过模型最大上下文长度
        2. 将token索引转换为嵌入向量
        3. 应用dropout（如果启用）
        4. 通过多层RWKV模块处理特征
            - 如果启用小注意力机制，使用特殊处理流程
            - 否则使用标准RWKV处理流程
        5. 应用最终层归一化
        6. 计算输出logits
            - 如果启用头注意力机制，添加额外的注意力输出
        7. 返回最终logits
        """
        args = self.args
        B, T = idx.size()  # 获取输入形状
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."  # 检查序列长度

        # 嵌入层
        x = self.emb(idx)  # 将token索引转换为嵌入向量
        x_emb = x  # 保存初始嵌入

        # 应用dropout
        if args.dropout > 0:
            x = self.drop0(x)

        # 处理小注意力机制
        if args.tiny_att_dim > 0:
            for block in self.blocks:
                if args.grad_cp == 1:  # 检查是否使用梯度检查点
                    x = deepspeed.checkpointing.checkpoint(block, x, x_emb)
                else:
                    x = block(x, x_emb)
        else:
            # 标准注意力处理
            v_first = torch.empty_like(x)  # 初始化值向量
            for block in self.blocks:
                if args.grad_cp == 1:
                    x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
                else:
                    x, v_first = block(x, v_first)

        # 最终层归一化
        x = self.ln_out(x)

        # 处理头注意力
        if args.head_qk > 0:
            q = self.head_q(x)[:, :T, :]  # 查询
            k = self.head_k(x)[:, :T, :]  # 键
            c = (q @ k.transpose(-2, -1)) * (1.0 / args.head_qk)  # 计算注意力分数：查询和键的点积，并缩放
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)  # 应用掩码

            # 根据浮点模式处理输出
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                c = c @ F.one_hot(idx, num_classes=args.vocab_size)
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                c = c @ F.one_hot(idx, num_classes=args.vocab_size).half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                c = c @ F.one_hot(idx, num_classes=args.vocab_size).bfloat16()

            x = self.head(x) + c  # 添加注意力输出
        else:
            x = self.head(x)  # 普通输出

        return x  # 返回最终logits

    def training_step(self, batch, batch_idx):
        args = self.args
        if args.my_qa_mask != 1:
            idx, targets = batch
            logits = self(idx)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))  # 计算交叉熵损失：预测logits与目标标签
        else:
            idx, targets, mask = batch
            mask = mask.view(-1)
            sum_mask = torch.sum(mask).item()

            logits = self(idx)
            if sum_mask == mask.shape[0]:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                # print('rank', self.global_rank, 'loss', loss.item())
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
                loss = torch.sum(loss * mask) / sum_mask

        return L2Wrap.apply(loss, logits)

    def training_step_end(self, batch_parts):
        if pl.__version__[0]!='2':
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all
#初始化
    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        n_params = 0
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            s3 = str(shape[3]) if len(shape) > 3 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}", end="")

            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n or n.endswith('_w') or n.endswith('_w1') or n.endswith('_w2') or n.endswith('_bias') or (".weight" not in n):
                if 'ln_x.weight' in n:
                    layer_scale = (1+int(n.split('.')[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
                print()
            elif n == "emb.weight":
                m[n] = p
                scale = -1e-4
                nn.init.uniform_(m[n], a=scale, b=-scale)
                print(f" [scale {scale}]")
            elif n == "head.weight":
                m[n] = p
                if self.args.vocab_size > self.args.n_embd:
                    scale = 0.5 * math.sqrt(self.args.vocab_size / self.args.n_embd)
                else:
                    scale = 0.5
                nn.init.orthogonal_(m[n], gain=scale)  # 使用正交初始化权重矩阵，保持输入输出特征之间的独立性
                print(f" [scale {scale}]")
            else:
                assert n.endswith('.weight') # should always be true

                zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']

                for kk in zero:
                    if kk in n:
                        scale = 0
                if "head_k." in n:
                    scale = 0.1
                if "head_q." in n:
                    scale = 0

                for kk in [".att.key."]:
                    if kk in n:
                        scale = 0.1
                for kk in [".att.gate."]:
                    if kk in n:
                        scale = 0.1

                print(f" [scale {scale}]")

                if self.args.accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=scale)

            m[n] = m[n].cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()
            n_params += m[n].numel()

        print('model params', n_params)
        gc.collect()
        torch.cuda.empty_cache()
        return m
