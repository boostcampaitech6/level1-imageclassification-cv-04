# ${변수} 정의
NAME="exp"
# TARGET_GENDER=male
# TARGET_MASK=2
WANDB="model_CLIPQA"
EPOCH=20
BATCH=512    # 256
DATASET="MaskSplitByProfileDataset"
DATA_USE=1
MODEL="CLIPQA"
# TODO: 2. model training
MODEL_ARCH=1
AUG="BaseAugmentation"
LOSS="f1"
OPT="Adam"
# TODO: 7. lr scheduler
# TODO: 8. additional
DATA_DIR="/data/ephemeral/home/maskdata"
OUTPUT_DIR="/data/ephemeral/home/output"
DECAY_STEP=5

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
--lr 0.001 \
--resize 224 224   # For CLIP model


# --valid_batch_size 100 \
# --target_gender ${TARGET_GENDER} \
# --target_mask ${TARGET_MASK} \