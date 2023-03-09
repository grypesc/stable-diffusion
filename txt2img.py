import time
import torch

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


@torch.no_grad()
def main():
    repo_id = "stabilityai/stable-diffusion-2"
    pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float32, revision="fp16")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    prompt = "Swag gangsters drifing cadillac and shooting from AK-47."
    negative_prompt = "Road is empty"
    pipe.enable_attention_slicing()
    images = pipe(prompt, num_inference_steps=25, negative_prompt=negative_prompt, num_images_per_prompt=2, guidance_scale=7.5).images
    for i, image in enumerate(images):
        image.save(f"images/{str(time.time())}_{i}.png")


if __name__ == "__main__":
    main()
