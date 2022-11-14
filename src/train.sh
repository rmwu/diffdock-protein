# This script should contain run-specific information
# like GPU id, batch size, etc.
# Everything else should be specified in config.yaml

NUM_FOLDS=5
SEED=0
CUDA=4
NUM_GPU=4
BATCH_SIZE=40

NAME="dips"
CONFIG="config/${NAME}.yaml"

SAVE_PATH="/data/scratch/rmwu/tmp-runs/glue/${NAME}"

echo $SAVE_PATH

python src/main.py \
    --mode "train" \
    --config_file $CONFIG \
    --run_name $NAME \
    --save_path $SAVE_PATH \
    --checkpoint_path $SAVE_PATH \
    --batch_size $BATCH_SIZE \
    --num_folds $NUM_FOLDS \
    --num_gpu $NUM_GPU \
    --gpu $CUDA --seed $SEED

