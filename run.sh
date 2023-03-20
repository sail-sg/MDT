pip install -e .
export OPENAI_LOGDIR=output_mdt_s2
NUM_GPUS=8

MODEL_FLAGS="--image_size 256 --mask_ratio 0.30 --decode_layer 2 --model MDT_S_2"
DIFFUSION_FLAGS="--diffusion_steps 1000"
TRAIN_FLAGS="--batch_size 32"
DATA_PATH=/dataset/imagenet

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS scripts/image_train.py --data_dir $DATA_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS