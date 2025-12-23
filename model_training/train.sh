# Default values
DATASET="kvasir"
BATCH_SIZE=32
EPOCHS=20
NUM_CLASSES=3
LR=0.0001
MODEL_NAME="resnet18"
TORCH_PATH="path to download network"
CHECKPOINT_DIR="path to save checkpoints"
TRAIN_DIR="path where training data is saved"
TEST_DIR="path where testing data is saved"
SEED=42
python3 model_training/train.py --model_name "$MODEL_NAME" --batch_size "$BATCH_SIZE" --epochs "$EPOCHS" --dataset "$DATASET" --num_classes "$NUM_CLASSES" --torch_path "$TORCH_PATH" --checkpoint_dir "$CHECKPOINT_DIR" --train_dir "$TRAIN_DIR" --test_dir "$TEST_DIR" --seed "$SEED"