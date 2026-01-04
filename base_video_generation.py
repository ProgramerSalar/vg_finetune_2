import os
import json
import torch
import numpy as np
import PIL
from PIL import Image
from IPython.display import HTML
from pipeline import PyramidDiTForVideoGeneration
from diffusers.utils import export_to_video

if __name__ == "__main__":

    
    
    
    # ... (Your setups) ...
    variant='diffusion_transformer_384p' 
    model_name = "pyramid_flux"
    model_path = "/content/drive/MyDrive/PATH/pyramid-flow-miniflux"
    model_dtype = 'bf16' 
    # model_dtype = 'fp32' 


    device = torch.device("cuda") # Define device object once
    torch.cuda.set_device(0)

    model = PyramidDiTForVideoGeneration(
        model_path,
        model_dtype,
        model_name=model_name,
        model_variant=variant,
    )

   


    model.vae.to(device)
    model.dit.to(device)
    model.text_encoder.to(device)
    model.vae.enable_tiling()


    if model_dtype == "bf16":
        torch_dtype = torch.bfloat16 
    elif model_dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    prompt = "A movie trailer featuring the adventures of the 30 year old space man wearing a red wool knitted motorcycle helmet, blue sky, salt desert, cinematic style, shot on 35mm film, vivid colors"
    # prompt = "An educational math lesson explaining the slope intercept form Y=mx+b on a whiteboard, high quality, clear text"
    # prompt="Now let's say if we have an equation that looks like this 2x + y is equal to."
    # prompt="Six numbers they're followed by important vocab words and concepts so that's what we're going to cover in this video and on the right hand side of your"


    # used for 384p model variant
    width = 640
    height = 384

    # used for 768p model variant
    # width = 1280
    # height = 768

    temp = 16   # temp in [1, 31] <=> frame in [1, 241] <=> duration in [0, 10s]
    # Noting that, for the 384p version, only supports maximum 5s generation (temp = 16)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=True if model_dtype != 'fp32' else False, dtype=torch_dtype):
        frames = model.generate(
            prompt=prompt,
            num_inference_steps=[20, 20, 20],
            video_num_inference_steps=[10, 10, 10],
            height=height,
            width=width,
            temp=temp,
            guidance_scale=7.0,         # The guidance for the first frame, set it to 7 for 384p variant
            video_guidance_scale=5.0,   # The guidance for the other video latent
            output_type="pil",
            save_memory=True,           # If you have enough GPU memory, set it to `False` to improve vae decoding speed
        )

    export_to_video(frames, "./text_to_video_sample.mp4", fps=24)
    
    
   