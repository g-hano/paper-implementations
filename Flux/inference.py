import torch
from einops import rearrange
from PIL import Image
import os
import uuid
import time

from conditioner import HFEmbedder
from sampling import denoise, get_noise, get_schedule, prepare, unpack
from utils import configs, load_ae, load_flow_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t5 = HFEmbedder("google/t5-v1_1-xxl", max_length=256, torch_dtype=torch.bfloat16).to(device)
clip = HFEmbedder("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16).to(device)
model = load_flow_model("flux-schnell", device=device)
ae = load_ae("flux-schnell", device=device)

width: int = 512
heigth: int = 512
num_steps: int = 4
guidance: float = 3.5
seed: int = 1 # set it to None if you want random images
prompt: str = "Cat in the middle of desert." 
#A pumpkin cat. A cat that is merged with a pumpkin. Make it ultra realistic and a humanoid creature with a full body picture. Make his ears look like the inside of a pumpkin.

start = time.perf_counter()
print("Image Generation Started")

noisy_image = get_noise(
    num_samples=1,
    height=heigth,
    width=width,
    device=device,
    dtype=torch.bfloat16,
    seed=seed
)

timesteps = get_schedule(
    num_steps=num_steps,
    image_seq_len=noisy_image.shape[-1] * noisy_image.shape[-2] // 4,
    shift=False
)

inp = prepare(t5=t5, clip=clip, img=noisy_image, prompt=prompt)

noisy_image = denoise(model=model, **inp, timesteps=timesteps, guidance=guidance)

# Decode latents to pixel space
noisy_image = unpack(noisy_image.float(), height=heigth, width=width)
with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    noisy_image = ae.decode(noisy_image)

# Convert to PIL Image
noisy_image = noisy_image.clamp(-1, 1).float()
noisy_image = rearrange(noisy_image[0], "c h w -> h w c")

img = Image.fromarray((127.5 * (noisy_image + 1.0)).cpu().byte().numpy())

end = time.perf_counter()
print(f"Image Generation Finished in {end-start} seconds")

filename = f"flux_generated/{uuid.uuid4()}.jpg"
os.makedirs(os.path.dirname(filename), exist_ok=True)
img.save(filename, format="jpeg", quality=95, subsampling=0)