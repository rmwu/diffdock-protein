# This script should contain run-specific information
# like GPU id, batch size, etc.
# Everything else should be specified in config.yaml

NUM_FOLDS=5  # number of seeds to try
SEED=0  # initial seed
CUDA=4  # will use GPUs from CUDA to CUDA + NUM_GPU - 1
NUM_GPU=4
BATCH_SIZE=32  # split across all GPUs

NAME="dips_esm"  # change to name of config file
CONFIG="config/${NAME}.yaml"

# you may save to your own directory
SAVE_PATH="/data/scratch/rmwu/tmp-runs/glue/${NAME}"

echo $SAVE_PATH

python src/main.py \
    --mode "train" \
    --config_file $CONFIG \
    --run_name $NAME \
    --save_path $SAVE_PATH \
    --batch_size $BATCH_SIZE \
    --num_folds $NUM_FOLDS \
    --num_gpu $NUM_GPU \
    --gpu $CUDA --seed $SEED
    #--checkpoint_path $SAVE_PATH \

# if you accidentally screw up and the model crashes
# you can restore training (including optimizer)
# by uncommenting --checkpoint_path

