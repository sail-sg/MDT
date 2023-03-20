pip install -e .
export OPENAI_LOGDIR=output_mdt_xl2
MODEL_FLAGS="--image_size 256 --mask_ratio 0.30 --decode_layer 2 --model MDT_XL_2"
DIFFUSION_FLAGS="--diffusion_steps 1000"
TRAIN_FLAGS="--batch_size 4"
DATA_PATH=/dataset/imagenet
NUM_NODE=8
GPU_PRE_NODE=8

python -m torch.distributed.launch --master_addr=$(hostname) --nnodes=$NUM_NODE --node_rank=$RANK --nproc_per_node=$GPU_PRE_NODE --master_port=$MASTER_PORT scripts/image_train.py --data_dir $DATA_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

   