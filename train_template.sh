# 사용 방법
# 각 head를 학습시키려면,
#       --target age / gender / mask 를 사용합니다.
#       MODEL="CLIP3Head3Proj"
#       EPOCH=10
#       LR=0.001
#       DECAY_STEP=5
# Aggregation 모델을 학습시키려면,
#       --target 인자를 제거합니다. 그냥 쓰지 말아요.
#       MODEL="CLIP3Head3Proj_Aggregation"
#       EPOCH=30
#       LR=0.0001
#       DECAY_STEP=10

# ${변수} 정의
NAME="test" # test / exp
WANDB="test_CLIP3Head3Proj_mask"
EPOCH=15
BATCH=512    # 경향을 파악하는건 64가 나은 듯
DATASET="MaskSplitByProfileDataset"
DATA_USE=1
MODEL="CLIP3Head3Proj"
# TODO: 2. model training
MODEL_ARCH=1
AUG="CustomAugmentation"
LR=0.001
SCHEDULER=StepLR
KFOLD=0
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
--criterion ${LOSS} \
--optimizer ${OPT} \
--lr_decay_step ${DECAY_STEP} \
--name ${NAME} \
--wandb ${WANDB} \
--data_dir "${DATA_DIR}/train/images/" \
--model_dir "${OUTPUT_DIR}" \
--val_ratio 0.2 \
--lr ${LR} \
--augmentation ${AUG} \
--scheduler ${SCHEDULER} \
--kfold ${KFOLD} \
--target mask \
--resize 224 224   # For CLIP model


# --valid_batch_size 378 \

# --target_gender ${TARGET_GENDER} \
# --target_mask ${TARGET_MASK} \