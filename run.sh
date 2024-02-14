pip3 install torch==2.0 torchvision torchaudio
pip install -e .
python -m pip install git+https://github.com/sail-sg/Adan.git
export OPENAI_LOGDIR=output_mdtv2_s2
NUM_GPUS=8

MODEL_FLAGS="--image_size 256 --mask_ratio 0.30 --decode_layer 4 --model MDTv2_S_2"
DIFFUSION_FLAGS="--diffusion_steps 1000"
TRAIN_FLAGS="--batch_size 32 --lr 5e-4"
DATA_PATH=/dataset/imagenet-raw/train

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS scripts/image_train.py --data_dir $DATA_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS