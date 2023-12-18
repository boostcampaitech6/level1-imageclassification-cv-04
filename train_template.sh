# ${변수} 정의
NAME="exp"
WANDB="model_EfficientNetB0MultiHead"
VAL_TAB=2
EPOCH=30
BATCH=512
DATASET="MaskSplitByProfileDataset"
DATA_USE=1
MODEL="EfficientNetB0MultiHead"
# TODO: 2. model training
MODEL_ARCH=0
AUG="BaseAugmentation"
LOSS="f1"
OPT="Adam"
# TODO: 7. lr scheduler
DECAY_STEP=5
# TODO: 8. additional
DATA_DIR="/data/ephemeral/home/level1-imageclassification-cv-04/data"
OUTPUT_DIR="/data/ephemeral/home/level1-imageclassification-cv-04/results"

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
--save_val_table ${VAL_TAB} \
--data_dir "${DATA_DIR}/train/images/" \
--model_dir "${OUTPUT_DIR}"