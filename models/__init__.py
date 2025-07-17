import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, StableDiffusionXLPipeline

from handlers import get_handler, FALSE_SA_ARGS
from models.sdxl_controlnet_balanced import StableDiffusionXLControlNetBalancedPipeline
from models.sdxl_balanced import StableDiffusionXLBalancedPipeline


def create_infer_model(pipeline, num_style_layers, controlnet_model=False, controlnet_removal_amount=0, balance_values=False, approximate_timesteps=False):
    handler_args = FALSE_SA_ARGS
    handler_cls = get_handler()

    if balance_values:
        style_stats = torch.load("grading_files/style_balance_grading_v", weights_only=False)
    else:
        style_stats = torch.load("grading_files/style_layer_grading", weights_only=False)
    handler = handler_cls(pipeline, style_stats, num_style_layers, approximate_timesteps)
    handler.register(handler_args)
    
    if controlnet_model:
        infer_pipeline = StableDiffusionXLControlNetBalancedPipeline(pipeline, handler, controlnet_removal_amount)
    else:
        infer_pipeline = StableDiffusionXLBalancedPipeline(pipeline, handler)

    print("===================== Inference Info =====================")
    print("Num Style Layers: {} (lambda_s=~{:.2f})".format(num_style_layers, num_style_layers/70))
    if controlnet_removal_amount > 0:
        print("ControlNet Removal Amount: {} (lambda_t=~{:.2f})".format(controlnet_removal_amount, controlnet_removal_amount/1000))
    print("=======================================================")
    return infer_pipeline


def load_controlnet(control_type, device=None):
    if control_type == "depth" or control_type is None:
        control_path = "diffusers/controlnet-depth-sdxl-1.0"
    elif control_type == "canny":
        control_path = "diffusers/controlnet-canny-sdxl-1.0"
    else:
        print(f"control type {control_type} is not supported.")
        exit(-1)

    control_net = ControlNetModel.from_pretrained(
        control_path,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )

    if device is not None:
        control_net = control_net.to(device)
    return control_net


def load_ddpm(controlnet=None, device=None):
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    if device is not None:
        vae = vae.to(device)

    if controlnet is None:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
        )
    else:
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        )

    if device is not None:
        pipeline = pipeline.to(device)

    pipeline.enable_model_cpu_offload()
    return pipeline

