# ${변수} 정의
# resize shape
RESIZE="
224
224
"
MODEL_ARCH=1
MODEL=SwinTransformerBase224
RESULT_NAME="SwinTransformerBase224_focal"
TEST_DIR="/data/ephemeral/home/level1-imageclassification-cv-04/data/eval"
MODEL_DIR="/data/ephemeral/home/level1-imageclassification-cv-04/results/${RESULT_NAME}/best.pth"

# run with args
python inference.py \
--resize ${RESIZE} \
--multi_head ${MODEL_ARCH} \
--model ${MODEL} \
--test_dir "${TEST_DIR}" \
--model_path "${MODEL_DIR}"