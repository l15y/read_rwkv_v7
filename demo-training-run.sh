#!/bin/bash
#######################################################################################################################
#
# 运行本脚本前请先运行demo-training-prepare.sh（使用相同的MODEL_TYPE、N_LAYER和N_EMBD参数）
# 或者将基础模型重命名为rwkv-init.pth并放入输出文件夹
#
# 训练器会加载文件夹中最后一个rwkv-*.pth文件，以便可以从停止的运行继续训练
# 因此请检查日志（### Loading rwkv-xxx.pth... ###），确保没有多余的rwkv-*.pth文件
#
#######################################################################################################################
#
# 模型结构参数
N_LAYER="12"  # 模型层数
N_EMBD="768"  # 嵌入维度
#
CTX_LEN="4096" # 上下文长度，注意：如果修改ctx_len需要同时修改magic_prime
PROJ_DIR="out" # 设置输出文件夹
#
#######################################################################################################################
#
# 注意：批次大小和学习率会影响模型和训练性能
# 小数据 => 使用较小的批次大小和稍小的学习率
# 大数据 => 使用较大的批次大小和稍大的学习率
# 较大模型 => 使用较小的学习率
# 微调 => 使用非常小的学习率，如1e-5
#
M_BSZ="32" # 此处约占用9G显存 => 减小此值以节省显存，增大此值以提高速度
LR_INIT="1e-5"  # 初始学习率
LR_FINAL="6e-6"  # 最终学习率
GRAD_CP=1 # 1 => 较慢，节省显存; 0 => 较快，占用更多显存
EPOCH_SAVE=1 # 每10个"mini epoch"保存一次（1 mini epoch = 40320 * ctx_len tokens）=> 如果GPU较弱则减小此值
#
#######################################################################################################################
#
# magic_prime = 小于 datalen/ctxlen-1 的最大3n+2质数（本例中为1498226207/512-1 = 2926222.06，magic_prime为2926181）
# 可以使用 https://www.dcode.fr/prime-numbers-search 查找
#
N_NODE=1 # 节点数量
GPU_PER_NODE=1 # 每个节点的GPU数量
#
DS_BUCKET_MB=200 # 消费级GPU设置为2，A100/H100设置为200（影响速度和显存使用）
#
# 运行训练脚本
python train.py --load_model "RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth" --wandb "Testv7" --proj_dir $PROJ_DIR \
 --ctx_len $CTX_LEN --my_pile_stage 3 --epoch_count 999999 --epoch_begin 0 \  # 训练阶段和epoch设置
 --data_file "data/minipile" --my_exit_tokens 1498226207 --magic_prime 365759 \  # 数据文件和训练终止条件
 --num_nodes $N_NODE --micro_bsz $M_BSZ --n_layer $N_LAYER --n_embd $N_EMBD \  # 分布式训练和模型结构参数
 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 --my_pile_edecay 0 --vocab_size 65536 \  # 学习率和优化器参数
 --weight_decay 0.001 --epoch_save $EPOCH_SAVE --head_size_a 64 \  # 权重衰减和模型保存参数
 --accelerator gpu --devices $GPU_PER_NODE --precision bf16 --strategy deepspeed_stage_2 --grad_cp $GRAD_CP --enable_progress_bar True --ds_bucket_mb $DS_BUCKET_MB  # 硬件和精度设置
