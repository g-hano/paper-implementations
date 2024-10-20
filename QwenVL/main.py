from modeling import *
from processor import Qwen2VLProcessor
from img_process import Qwen2VLImageProcessor

from transformers import AutoConfig, AutoTokenizer
import torch
from safetensors import safe_open
from PIL import Image

tokenizer = AutoTokenizer.from_pretrained("paper-implementations/qwen2.5-vl/Qwen2VL")
config = AutoConfig.from_pretrained("paper-implementations/qwen2.5-vl/Qwen2VL/config.json")
generation_config = AutoConfig.from_pretrained("paper-implementations/qwen2.5-vl/Qwen2VL/generation_config.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

image_processor = Qwen2VLImageProcessor()
processor = Qwen2VLProcessor(image_processor=image_processor, tokenizer=tokenizer, chat_template=tokenizer.chat_template)

import os, gc
def load_model_weights(model, folder_path, device="cuda", dtype=torch.bfloat16):
    # Ensure model is on CPU and clear CUDA cache
    model = model.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.safetensors')])
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        print(f"Loading file: {file_path}")

        # Load the state dict from the safetensors file
        with safe_open(file_path, framework="pt") as f:
            state_dict = {k: f.get_tensor(k).clone().detach() for k in f.keys()}
        
        # Load the state dict into the model
        model.load_state_dict(state_dict, strict=False)  # strict=False to allow partial weights

    print("All files loaded successfully.")
    return model.to(device=device, dtype=dtype)


model = Qwen2VLForConditionalGeneration(config=config, generation_config=generation_config)
model = load_model_weights(model, r"paper-implementations\qwen2.5-vl\Qwen2VL")
model.eval()

print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
print(f"Max allocated GPU memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

print_details = True
if print_details:
	first_param = next(model.parameters())
	print(f"Model parameters dtype: {first_param.dtype}")
	print(f"Model is on CUDA: {next(model.parameters()).is_cuda}")
	print(f"visual.patch_embed.proj weight dtype: {model.visual.patch_embed.proj.weight.dtype}")
	print(f"model.embed_tokens weight dtype: {model.model.embed_tokens.weight.dtype}")
	print(f"lm_head weight dtype: {model.lm_head.weight.dtype}")
	print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
	print(f"Max allocated GPU memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

image = Image.open("C:/Users/Cihan/Desktop/llamaindex/downloaded_image.jpg")
prompt = "Describe this image."
conversation = [
	{
		"role": "user",
		"content": [
			{
				"type": "image",
			},
			{
				"type": "text", 
				"text": prompt
			},
		],
	}
]
text_prompt = tokenizer.apply_chat_template(conversation, 
                                            add_generation_prompt=True, 
                                            tokenize=False)

inputs = processor(
	text=[text_prompt], images=[image]
)
for k, v in inputs.items():
    print(f"{k}: {v.shape}")

output_ids = model(**inputs)

generated_ids = [
	output_ids[len(input_ids) :]
	for input_ids, output_ids in zip(inputs.input_ids, output_ids)
]
output_text = image_processor.batch_decode(
	generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
print(output_text)

print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
print(f"Max allocated GPU memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")