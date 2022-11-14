# main script for training

# training params
SEED=0
CUDA=6
BATCH_SIZE=32

NAME="db5_denoise"
CONFIG="config/${NAME}.json"
CHECKPOINT="/data/scratch/rmwu/tmp-runs/ml-energy/db5_denoise-pos/"
SAVE_PATH=$CHECKPOINT
TEST_FOLD=4

echo $SAVE_PATH

python src/main.py \
    --mode "test" \
    --config_file $CONFIG \
    --run_name $NAME \
    --save_path $SAVE_PATH \
    --checkpoint_path $CHECKPOINT \
    --batch_size $BATCH_SIZE \
    --test_fold $TEST_FOLD \
    --gpu $CUDA --seed $SEED \
    #--use_unbound \

