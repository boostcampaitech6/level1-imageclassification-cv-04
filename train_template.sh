# ${변수} 정의
NAME="exp"
WANDB="model_CLIP3Head3Proj_Aggregation"
EPOCH=20
BATCH=512    # 경향을 파악하는건 64가 나은 듯
DATASET="MaskSplitByProfileDataset"
DATA_USE=1
MODEL="CLIP3Head3Proj_Aggregation"
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