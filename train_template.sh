# ${변수} 정의
NAME="reboot"
WANDB="test_CLIP3Head3Proj_Aggregation2"
EPOCH=30
BATCH=256    # 경향을 파악하는건 64가 나은 듯
DATASET="MaskSplitByProfileDataset"
DATA_USE=1
MODEL="CLIP3Head3Proj_Aggregation2"
# TODO: 2. model training
MODEL_ARCH=1
AUG="BaseAugmentation"
LOSS="f1"
OPT="Adam"
# TODO: 7. lr scheduler
# TODO: 8. additional
DATA_DIR="/data/ephemeral/home/maskdata"
OUTPUT_DIR="/data/ephemeral/home/output"
DECAY_STEP=10

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
--val_ratio 0.1 \
--lr 0.0001 \
--resize 224 224   # For CLIP model


# --valid_batch_size 100 \
# --target_gender ${TARGET_GENDER} \
# --target_mask ${TARGET_MASK} \