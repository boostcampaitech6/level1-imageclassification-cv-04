# ${변수} 정의
NAME="exp"
EPOCH=30
BATCH=512
DATASET="MaskSplitByProfileDataset"
DATA_USE=True
MODEL="EfficientNetB0MultiHead"
# TODO: 2. model training
MODEL_ARCH=True
AUG="BaseAugmentation"
LOSS="f1"
OPT="Adam"
# TODO: 7. lr scheduler
# TODO: 8. additional
DATA_DIR="/data/ephemeral/home/level1-imageclassification-cv-04/data"
OUTPUT_DIR="/data/ephemeral/home/level1-imageclassification-cv-04/code/v3/results"

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
--name ${NAME} \
--data_dir "${DATA_DIR}/train/images/" \
--model_dir "${OUTPUT_DIR}"