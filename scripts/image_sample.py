"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from masked_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
    args_to_dict,
)

from masked_diffusion import (
        create_diffusion,
        model_and_diffusion_defaults,
        diffusion_defaults, 
        dist_util,
        logger,
)

import masked_diffusion.models as models_mdt
from diffusers.models import AutoencoderKL

def main():
    th.backends.cuda.matmul.allow_tf32 = True
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")

    configs = args_to_dict(args, model_and_diffusion_defaults().keys())
    print(configs)
    image_size = configs['image_size']
    latent_size = image_size // 8
    model = models_mdt.__dict__[args.model](input_size=latent_size, decode_layer=args.decode_layer)
    msg = model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    print(msg)
    config_diffusion = args_to_dict(args, diffusion_defaults().keys())
    config_diffusion['timestep_respacing']= str(args.num_sampling_steps)
    print(config_diffusion)
    diffusion = create_diffusion(**config_diffusion)
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    th.set_grad_enabled(False)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-"+str(args.vae_decoder)).to(dist_util.dev())

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.cfg_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            z = th.randn(args.batch_size, 4, latent_size, latent_size, device=dist_util.dev())
            # Setup classifier-free guidance:
            z = th.cat([z, z], 0)
            classes_null = th.tensor([NUM_CLASSES] * args.batch_size, device=dist_util.dev())
            classes_all = th.cat([classes, classes_null], 0)
            model_kwargs["y"] = classes_all
            model_kwargs["cfg_scale"] = args.cfg_scale
            model_kwargs["diffusion_steps"] = config_diffusion['diffusion_steps']
            model_kwargs["scale_pow"] = args.scale_pow
        else:
            z = th.randn(args.batch_size, 4, latent_size, latent_size, device=dist_util.dev())
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes


        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=args.clip_denoised,
            progress=True, 
            model_kwargs=model_kwargs,
            device=dist_util.dev()
        )
        if args.cfg_cond:
            sample, _ = sample.chunk(2, dim=0)  # Remove null class samples
        # latent to image
        sample = vae.decode(sample / 0.18215).sample
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8) # clip in range -1,1
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        num_sampling_steps=250,
        clip_denoised=False,
        num_samples=5000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        model="MDT_S_2",
        class_cond=True,
        cfg_scale=3.8,
        decode_layer=None,
        cfg_cond=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale_pow', default=4, type=float)
    parser.add_argument('--vae_decoder', type=str, default='ema')  # ema or mse
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument(
        "--rank", default=0, type=int, help="""rank for distrbuted training."""
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
