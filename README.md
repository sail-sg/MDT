# Masked Diffusion Transformer V2

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/masked-diffusion-transformer-is-a-strong/image-generation-on-imagenet-256x256)](https://paperswithcode.com/sota/image-generation-on-imagenet-256x256?p=masked-diffusion-transformer-is-a-strong)
[![HuggingFace space](https://img.shields.io/badge/🤗-HuggingFace%20Space-cyan.svg)](https://huggingface.co/spaces/shgao/MDT)

The official codebase for [Masked Diffusion Transformer is a Strong Image Synthesizer](https://arxiv.org/abs/2303.14389).

## MDTv2: Faster Convergeence & Stronger performance
**MDTv2 demonstrates new State of the Art performance and a 5x acceleration compared to the original MDT.**

[MDTv1 code](https://github.com/sail-sg/MDT/tree/mdtv1)
## Introduction

Despite its success in image synthesis, we observe that diffusion probabilistic models (DPMs) often lack contextual reasoning ability to learn the relations among object parts in an image, leading to a slow learning process. To solve this issue, we propose a Masked Diffusion Transformer (MDT) that introduces a mask latent modeling scheme to explicitly enhance the DPMs’ ability to contextual relation learning among object semantic parts in an image. 

During training, MDT operates in the latent space to mask certain tokens. Then, an asymmetric diffusion transformer is designed to predict masked tokens from unmasked ones while maintaining the diffusion generation process. Our MDT can reconstruct the full information of an image from its incomplete contextual input, thus enabling it to learn the associated relations among image tokens. We further improve MDT with a more efficient macro network structure and training strategy, named MDTv2. 

Experimental results show that MDTv2 achieves superior image synthesis performance, e.g., **a new SOTA FID score of 1.58 on the ImageNet dataset, and has more than 10× faster learning speed than the previous SOTA DiT**. 

<img width="800" alt="image" src="figures/vis.jpg">

# Performance

| Model| Dataset |  Resolution | FID-50K | Inception Score |
|---------|----------|-----------|---------|--------|
|MDT-XL/2 | ImageNet | 256x256   | 1.79    | 283.01|
|MDTv2-XL/2 | ImageNet | 256x256 | 1.58    | 314.73|

[Pretrained model download](https://huggingface.co/shgao/MDT-XL2/tree/main)

Model is hosted on hugglingface, you can also download it with:
```
from huggingface_hub import snapshot_download
models_path = snapshot_download("shgao/MDT-XL2")
ckpt_model_path = os.path.join(models_path, "mdt_xl2_v1_ckpt.pt")
```
A hugglingface demo is on [DEMO](https://huggingface.co/spaces/shgao/MDT).

**NEW SOTA on FID.**
# Setup

Prepare the Pytorch 1.13 version. Download and install this repo.

```
git clone https://github.com/sail-sg/MDT
cd MDT
pip install -e .
```

**DATA** 
- For standard datasets like ImageNet and CIFAR, please refer to '[dataset](https://github.com/sail-sg/MDT/tree/main/datasets)' for preparation.
- When using customized dataset, change the image file name to `ClassID_ImgID.jpg`,
as the [ADM's dataloder](https://github.com/openai/guided-diffusion) gets the class ID from the file name. 

# Training

<details>
  <summary>Training on one node (`run.sh`). </summary>

```shell
export OPENAI_LOGDIR=output_mdtv2_s2
NUM_GPUS=8

MODEL_FLAGS="--image_size 256 --mask_ratio 0.30 --decode_layer 4 --model MDTv2_S_2"
DIFFUSION_FLAGS="--diffusion_steps 1000"
TRAIN_FLAGS="--batch_size 32"
DATA_PATH=/dataset/imagenet

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS scripts/image_train.py --data_dir $DATA_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

</details>

<details>
  <summary>Training on multiple nodes (`run_ddp_master.sh` and `run_ddp_worker.sh`). </summary>

```shell
# On master:
export OPENAI_LOGDIR=output_mdtv2_xl2
MODEL_FLAGS="--image_size 256 --mask_ratio 0.30 --decode_layer 2 --model MDTv2_XL_2"
DIFFUSION_FLAGS="--diffusion_steps 1000"
TRAIN_FLAGS="--batch_size 4"
DATA_PATH=/dataset/imagenet
NUM_NODE=8
GPU_PRE_NODE=8

python -m torch.distributed.launch --master_addr=$(hostname) --nnodes=$NUM_NODE --node_rank=$RANK --nproc_per_node=$GPU_PRE_NODE --master_port=$MASTER_PORT scripts/image_train.py --data_dir $DATA_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

# On workers:
export OPENAI_LOGDIR=output_mdtv2_xl2
MODEL_FLAGS="--image_size 256 --mask_ratio 0.30 --decode_layer 2 --model MDTv2_XL_2"
DIFFUSION_FLAGS="--diffusion_steps 1000"
TRAIN_FLAGS="--batch_size 4"
DATA_PATH=/dataset/imagenet
NUM_NODE=8
GPU_PRE_NODE=8

python -m torch.distributed.launch --master_addr=$MASTER_ADDR --nnodes=$NUM_NODE --node_rank=$RANK --nproc_per_node=$GPU_PRE_NODE --master_port=$MASTER_PORT scripts/image_train.py --data_dir $DATA_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS


```

</details>

# Evaluation

The evaluation code is obtained from [ADM's TensorFlow evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations).
Please follow the instructions in the `evaluations` folder to set up the evaluation environment.

<details>
  <summary>Sampling and Evaluation (`run_sample.sh`): </summary>

```shell
MODEL_PATH=output_mdtv2_xl2/mdt_xl2_v2_ckpt.pt
export OPENAI_LOGDIR=output_mdtv2_xl2_eval
NUM_GPUS=8

echo 'CFG Class-conditional sampling:'
MODEL_FLAGS="--image_size 256 --model MDTv2_XL_2 --decode_layer 4"
DIFFUSION_FLAGS="--num_sampling_steps 250 --num_samples 50000  --cfg_cond True"
echo $MODEL_FLAGS
echo $DIFFUSION_FLAGS
echo $MODEL_PATH
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS scripts/image_sample.py --model_path $MODEL_PATH $MODEL_FLAGS $DIFFUSION_FLAGS
echo $MODEL_FLAGS
echo $DIFFUSION_FLAGS
echo $MODEL_PATH
python evaluations/evaluator.py ../dataeval/VIRTUAL_imagenet256_labeled.npz $OPENAI_LOGDIR/samples_50000x256x256x3.npz

echo 'Class-conditional sampling:'
MODEL_FLAGS="--image_size 256 --model MDTv2_XL_2 --decode_layer 4"
DIFFUSION_FLAGS="--num_sampling_steps 250 --num_samples 50000"
echo $MODEL_FLAGS
echo $DIFFUSION_FLAGS
echo $MODEL_PATH
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS scripts/image_sample.py --model_path $MODEL_PATH $MODEL_FLAGS $DIFFUSION_FLAGS
echo $MODEL_FLAGS
echo $DIFFUSION_FLAGS
echo $MODEL_PATH
python evaluations/evaluator.py ../dataeval/VIRTUAL_imagenet256_labeled.npz $OPENAI_LOGDIR/samples_50000x256x256x3.npz
```

</details>

# Visualization

Run the `infer_mdt.py` to generate images.

# Citation

```
@misc{gao2023masked,
      title={Masked Diffusion Transformer is a Strong Image Synthesizer}, 
      author={Shanghua Gao and Pan Zhou and Ming-Ming Cheng and Shuicheng Yan},
      year={2023},
      eprint={2303.14389},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Acknowledgement

This codebase is built based on the [DiT](https://github.com/facebookresearch/dit) and [ADM](https://github.com/openai/guided-diffusion). Thanks!
