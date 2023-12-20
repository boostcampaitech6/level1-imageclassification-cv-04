# ${변수} 정의
SEED=42
KFOLD=0
EPOCHS=30
DATASET="MaskSplitByProfileDataset"
AUG="BaseAugmentation"
# BaseAugmentation / CustomAugmentation
CUTMIX=0
RESIZE="224 224"
TRAIN_BATCH=512
VALID_BATCH=1000
OPTIM="Adam"
# Adam / AdamW / NAdam / RAdam
LR=0.001
VALID_RATIO=0.2
LOSS="f1"
# ce / f1 / focal / smoothing
WEIGHT="max"
BEST_MODEL="acc"
SCHEDULER="StepLR"
DECAY_STEP=10
DECAY_RATE=0.5
PATIENCE=5
MODEL_ARCH=1  # 0: single-head, 1: multi-head
DATA_USE=1

MODEL="EfficientNetB0MultiHead"
# EfficientNetB0MultiHead / SwinTransformerBase224V1 / SwinTransformerBase224V2 / ResNet15 / EfficientNetB0MultiHead
RESULT_NAME="EfficientNetB0MultiHead_loss_weight"
WANDB_TYPE="test"
WANDB_CASE="EfficientNetB0MultiHead_loss_weight"
VALID_TABLE=0  # 0: not save, 1: save total validset, 2: save error case only
DATA_DIR="/data/ephemeral/home/level1-imageclassification-cv-04/data"
OUTPUT_DIR="/data/ephemeral/home/level1-imageclassification-cv-04/results"

## (+ early stopping 필수)
## my SOTA: model = swin transfer, augmentation = , loss = f1, optim = , lr = , lr decay step = , batch size = 

# run with args
python train.py \
--seed ${SEED} \
--epochs ${EPOCHS} \
--kfold ${KFOLD} \
--dataset ${DATASET} \
--augmentation ${AUG} \
--cutmix ${CUTMIX} \
--resize ${RESIZE} \
--batch_size ${TRAIN_BATCH} \
--valid_batch_size ${VALID_BATCH} \
--optimizer ${OPTIM} \
--lr ${LR} \
--val_ratio ${VALID_RATIO} \
--criterion ${LOSS} \
--weight ${WEIGHT} \
--best_model ${BEST_MODEL} \
--scheduler ${SCHEDULER} \
--lr_decay_step ${DECAY_STEP} \
--lr_decay_rate ${DECAY_RATE} \
--patience ${PATIENCE} \
--multi_head ${MODEL_ARCH} \
--use_caution_data ${DATA_USE} \
--model ${MODEL} \
--name ${RESULT_NAME} \
--wandb ${WANDB_TYPE}_${WANDB_CASE} \
--save_val_table ${VALID_TABLE} \
--data_dir "${DATA_DIR}/train/images/" \
--model_dir "${OUTPUT_DIR}"