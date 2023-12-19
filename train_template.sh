# ${변수} 정의
NAME="exp"
WANDB="model_EfficientNetB0MultiHead"
VAL_TAB=1
EPOCH=30
BATCH=512
DATASET="MaskSplitByProfileDataset"
DATA_USE=1
MODEL="EfficientNetB0MultiHead"
# TODO: 2. model training
MODEL_ARCH=0
AUG="CutmixAugmentation"
CUTMIX=1
LOSS="f1"
OPT="Adam"
LR=1e-3
# TODO: 7. lr scheduler
SCHEDULER="ReduceLROnPlateau"
DECAY_STEP=10   # for "StepLR"
DECAY_RATE=0.5  # for "StepLR"
PATIENCE=5     # for "ReduceLROnPlateau"
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
--cutmix ${CUTMIX} \
--criterion ${LOSS} \
--optimizer ${OPT} \
--lr ${LR} \
--scheduler ${SCHEDULER} \
--lr_decay_step ${DECAY_STEP} \
--lr_decay_rate ${DECAY_RATE} \
--patience ${PATIENCE} \
--name ${WANDB} \
--wandb ${WANDB} \
--save_val_table ${VAL_TAB} \
--data_dir "${DATA_DIR}/train/images/" \
--model_dir "${OUTPUT_DIR}"