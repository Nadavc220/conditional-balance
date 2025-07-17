""" 
A script for generating style aligned artistic images by applying the Conditional-Balanced method
over the StyleAligned method using a content image with depth based control-net.
"""

import os
import argparse

import torch
from PIL import Image

from utils import generate_file_name, get_depth_ctrl, load_control_imgs, initialize_latents_controlnet, save_control_imgs, prepare_target_prompts
from models import create_infer_model, load_controlnet, load_ddpm


NUM_TIMESTEPS = 1000
NUM_SELF_ATT_LAYERS = 70
CONTROL_TYPE = 'depth'

def main(args):
    device = 'cuda'

    # Load DDPM and Handler
    controlnet = load_controlnet(CONTROL_TYPE, device=device)
    pipeline = load_ddpm(controlnet, device=device)
    infer_pipeline = create_infer_model(pipeline, num_style_layers=args.num_style_layers, controlnet_model=True, controlnet_removal_amount=args.controlnet_removal_amount,
                                        balance_values=args.balance_values, approximate_timesteps=args.approximate_timesteps)
    
    # Init output path
    args.output_path = os.path.join(args.output_path, CONTROL_TYPE, str(args.seed))
    os.makedirs(args.output_path, exist_ok=True)

    # Generate Control Image
    controlnet_conditioning_scale = 0.8
    if args.control_image_path:
        control_img = load_control_imgs(args.control_image_path)
    else:
        control_img = get_depth_ctrl(args.content_image_path, device)
    num_controls = len(control_img) if type(control_img) is list else 1

    if not args.no_control_output:
        save_control_imgs(control_img, args.output_path)
    

    # Initialize DDPM parameters
    torch.manual_seed(args.seed)
    target_prompts = [f"{content_prompt}, {args.style_prompt}" for content_prompt in args.content_prompts]
    target_prompts = prepare_target_prompts(target_prompts, num_controls)
    reference_prompt = f"{args.reference_prompt}, {args.style_prompt}"
    prompts = [reference_prompt] + target_prompts

    latents = None
    if args.initialize_latents:
        latents = initialize_latents_controlnet(args.num_images_per_prompt, num_controls, pipeline.unet.dtype)

    # Run Pipeline
    images = infer_pipeline(prompts,
                            control_image=control_img,
                            num_inference_steps=args.num_inference_steps,
                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                            num_images_per_prompt=args.num_images_per_prompt,
                            latents=latents)
    
    # Save Output
    f_name = generate_file_name(args.num_style_layers, controlnet_removal_amount=args.controlnet_removal_amount, content_image_path=None, seed=args.seed)
    images[0].save(os.path.join(args.output_path, f"ref_{args.seed}.png"))
    for i in range(0, len(images) - 1):
        images[i+1].save(os.path.join(args.output_path, f"{i}_" + f_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed")
    parser.add_argument('--content_prompts', nargs='+', type=str, help='List of prompts to generate')
    parser.add_argument("--style_prompt", type=str)
    parser.add_argument("--reference_prompt", type=str)
    parser.add_argument("--content_image_path", type=str, default=None)
    parser.add_argument('--control_image_path', default=None)
    parser.add_argument('--no_control_output', action="store_true")
    parser.add_argument('--lambda_s', default=0.43, type=float)
    parser.add_argument('--lambda_t', default=0.85, type=float)
    parser.add_argument("--balance_values", action='store_true')
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--initialize_latents", action='store_true')
    parser.add_argument("--approximate_timesteps", action='store_true')
    parser.add_argument('--output_path')
    args = parser.parse_args()
    
    assert 0 <= args.lambda_s <= NUM_SELF_ATT_LAYERS, "Plase enter a float value in range [0, 1] (ratio) or an int value in range [0, 70] (number of layers for stylization)"
    if 0 < args.lambda_s <= 1:
        args.lambda_s = args.lambda_s * NUM_SELF_ATT_LAYERS
    args.num_style_layers = int(args.lambda_s)

    assert 0 <= args.lambda_t <= NUM_TIMESTEPS, "Plase enter a float value in range [0, 1] for setting the geometric style ration (0-no geometric style, 1-full geometric style)"
    args.controlnet_removal_amount = int(args.lambda_t * NUM_TIMESTEPS)

    assert (args.content_image_path is None) ^ (args.control_image_path is None), "Please provide exactly one of the following params: content_image_path, control_image_path"
    main(args)

