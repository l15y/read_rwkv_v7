# RWKV V7 模型训练项目

本项目是基于RWKV V7模型的开源实现，专注于模型训练和优化。项目代码主要从以下两个仓库提取并优化：
- [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM)
- [Triang-jyed-driung/rwkv7mini](https://github.com/Triang-jyed-driung/rwkv7mini)

## 项目特点
- 精简的模型实现，专注于V7版本
- 详细的代码注释，便于理解和修改
- 支持CUDA加速
- 提供完整的训练流程

## 环境要求
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (如需GPU加速)
- NVIDIA显卡 (如需GPU加速)

## 安装步骤
1. 克隆本仓库
   ```bash
   git clone https://github.com/your-repo/rwkv-v7.git
   cd rwkv-v7
   ```

2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

3. 编译CUDA扩展 (可选)
   ```bash
   cd cuda
   python setup.py install
   ```

## 文件结构
```
.
├── binidx.py            # 二进制索引处理
├── dataset.py           # 数据集处理
├── model.py             # RWKV V7模型实现
├── train.py             # 训练主程序
├── trainer.py           # 训练器实现
├── utils.py             # 工具函数
├── demo-training-prepare.sh  # 训练准备脚本
├── demo-training-run.sh      # 训练运行脚本
└── cuda/                # CUDA实现
    ├── wkv7_cuda.cu     # CUDA内核
    └── wkv7_op.cpp      # CUDA操作符
```

## 使用说明
1. 准备数据
   ```bash
   bash demo-training-prepare.sh
   ```

2. 开始训练
   ```bash
   bash demo-training-run.sh
   ```

   或直接运行：
   ```bash
   python train.py
   ```

## 贡献指南
欢迎提交PR或issue。请确保：
- 代码风格一致
- 添加必要的注释
- 更新相关文档