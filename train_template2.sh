# `bash train_template.sh model_A A BaseAugmentation 5 0.001`

WANDB=$1        # "model_A"
MODEL=$2        # "A"
AUG=$3          # "BaseAugmentation"
DECAY_STEP=$4   # 5
LR=$5           # 0.001

# ${변수} 정의
NAME="exp"
BATCH=512
DATASET="MaskSplitByProfileDataset"
DATA_USE=1
MODEL_ARCH=1
OPT="Adam"
DATA_DIR="/data/ephemeral/home/maskdata"
OUTPUT_DIR="/data/ephemeral/home/output"

for EPOCH in 20 30
do
    for LOSS in f1 cross_entropy focal
    do
        # run with args
        python train.py \
        --epochs ${EPOCH} \
        --batch_size ${BATCH} \
        --dataset ${DATASET} \
        --use_caution_data ${DATA_USE} \
        --model ${MODEL} \
        --multi_head ${MODEL_ARCH} \
        --augmentation ${AUG} \
        --criterion ${LOSS} \
        --optimizer ${OPT} \
        --lr_decay_step ${DECAY_STEP} \
        --name ${NAME} \
        --wandb ${WANDB} \
        --data_dir "${DATA_DIR}/train/images/" \
        --model_dir "${OUTPUT_DIR}" \
        --lr ${LR} &&   
    done
done


# 또는 아래처럼 연달아 작성해도 ㄱㅊ

# python train.py \
# --epochs ${EPOCH} \
# --batch_size ${BATCH} \
# --dataset ${DATASET} \
# --use_caution_data ${DATA_USE} \
# --model ${MODEL} \
# --multi_head ${MODEL_ARCH} \
# --augmentation ${AUG} \
# --criterion ${LOSS} \
# --optimizer ${OPT} \
# --lr_decay_step ${DECAY_STEP} \
# --name ${NAME} \
# --wandb ${WANDB} \
# --data_dir "${DATA_DIR}/train/images/" \
# --model_dir "${OUTPUT_DIR}" \
# --lr ${LR} && \
# python train.py \
# --epochs ${EPOCH} \
# --batch_size ${BATCH} \
# --dataset ${DATASET} \
# --use_caution_data ${DATA_USE} \
# --model ${MODEL} \
# --multi_head ${MODEL_ARCH} \
# --augmentation ${AUG} \
# --criterion ${LOSS} \
# --optimizer ${OPT} \
# --lr_decay_step ${DECAY_STEP} \
# --name ${NAME} \
# --wandb ${WANDB} \
# --data_dir "${DATA_DIR}/train/images/" \
# --model_dir "${OUTPUT_DIR}" \
# --lr ${LR}