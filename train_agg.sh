# `bash train_template.sh model_A A BaseAugmentation 5 0.001`

WANDB=model_CLIP3Head3Proj_Aggregation
MODEL=CLIP3Head3Proj_Aggregation       
AUG=BaseAugmentation         
DECAY_STEP=10       
LR=0.0001           
EPOCH=30
LOSS=f1
NAME="lastpang" # NOTE!
BATCH=512
DATASET="MaskSplitByProfileDataset"
DATA_USE=1
MODEL_ARCH=1
OPT="Adam"
DATA_DIR="/data/ephemeral/home/maskdata"
OUTPUT_DIR="/data/ephemeral/home/output"

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
--lr ${LR} \
--resize 224 224

