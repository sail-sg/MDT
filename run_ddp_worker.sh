pip3 install torch==2.0 torchvision torchaudio
python -m pip install git+https://github.com/sail-sg/Adan.git
pip install -e .
export OPENAI_LOGDIR=output_mdtv2_xl2
MODEL_FLAGS="--image_size 256 --mask_ratio 0.30 --decode_layer 4 --model MDTv2_XL_2"
DIFFUSION_FLAGS="--diffusion_steps 1000"
TRAIN_FLAGS="--batch_size 8"
DATA_PATH=/dataset/imagenet-raw/train
NUM_NODE=4
GPU_PRE_NODE=8

python -m torch.distributed.launch --master_addr=$MASTER_ADDR --nnodes=$NUM_NODE --node_rank=$RANK --nproc_per_node=$GPU_PRE_NODE --master_port=$MASTER_PORT scripts/image_train.py --data_dir $DATA_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
