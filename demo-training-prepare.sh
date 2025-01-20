#!/bin/bash
#######################################################################################################################
#
# 本脚本用于生成初始模型并保存到输出文件夹
#
#######################################################################################################################
#
# 请先创建data文件夹并下载minipile数据集（1498226207个token，约3GB）
mkdir -p data
# wget --continue -O data/minipile.idx https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx
# wget --continue -O data/minipile.bin https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin
#
#######################################################################################################################
#
# 模型结构参数
N_LAYER="12"  # 模型层数
N_EMBD="768"  # 嵌入维度
#
CTX_LEN="512" # 上下文长度，注意：如果修改ctx_len需要同时修改magic_prime
PROJ_DIR="out/L"$N_LAYER"-D"$N_EMBD"-"$MODEL_TYPE # 设置输出文件夹
#
#######################################################################################################################
#
# magic_prime = 小于 datalen/ctxlen-1 的最大3n+2质数（本例中为1498226207/512-1 = 2926222.06，magic_prime为2926181）
# 可以使用 https://www.dcode.fr/prime-numbers-search 查找
#
# 运行训练脚本
python train.py --wandb "" --proj_dir $PROJ_DIR \
 --data_file "data/minipile" --vocab_size 65536 \  # 数据文件和词汇表大小
 --ctx_len $CTX_LEN --my_pile_stage 1 --epoch_count 1 --epoch_begin 0 \  # 训练阶段和epoch设置
 --epoch_save 1 --weight_decay 0 --head_size_a 64 \  # 模型保存和优化器参数
 --num_nodes 1 --micro_bsz 1 --n_layer $N_LAYER --n_embd $N_EMBD --my_exit_tokens 1498226207 --magic_prime 365759 \  # 分布式训练和模型结构参数
 --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --my_pile_edecay 0 \  # 学习率和优化器参数
 --accelerator cpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 1  # 硬件和精度设置
