# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchvision.utils import save_image
from masked_diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from masked_diffusion.models import MDT_XL_2


# Setup PyTorch:
torch.manual_seed(0)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
num_sampling_steps = 500
cfg_scale = 5.0
pow_scale = 0.01 # large pow_scale increase the diversity, small pow_scale increase the quality.
model_path = 'mdt_xl2_v1_ckpt.pt'

# Load model:
image_size = 256
assert image_size in [256], "We provide pre-trained models for 256x256 resolutions for now."
latent_size = image_size // 8
model = MDT_XL_2(input_size=latent_size, decode_layer=2).to(device)

state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict)
model.eval()
diffusion = create_diffusion(str(num_sampling_steps))
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

# Labels to condition the model with:
class_labels = [208]*3

# Create sampling noise:
n = len(class_labels)
z = torch.randn(n, 4, latent_size, latent_size, device=device)
y = torch.tensor(class_labels, device=device)

# Setup classifier-free guidance:
z = torch.cat([z, z], 0)
y_null = torch.tensor([1000] * n, device=device)
y = torch.cat([y, y_null], 0)


model_kwargs = dict(y=y, cfg_scale=cfg_scale, scale_pow=pow_scale)

# Sample images:
samples = diffusion.p_sample_loop(
    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
)
samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
samples = vae.decode(samples / 0.18215).sample

# Save and display images:
save_image(samples, "sample.jpg", nrow=3, normalize=True, value_range=(-1, 1))
