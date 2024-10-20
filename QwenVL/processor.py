from transformers.processing_utils import ProcessorMixin, ProcessingKwargs
import torch
def to_tensor(data, device):
    if isinstance(data, list):
        data = torch.tensor(data)
    if torch.is_tensor(data):
        data = data.to(device)
    return data

class Qwen2VLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }

class Qwen2VLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    
    def __init__(self, image_processor, tokenizer, chat_template, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.device = device

    def __call__(self, images=None, text=None, videos=None, **kwargs):
        """
        Returns:
        - input_ids => List of token ids to be fed to model
        - attention_mask => List of indices specifying which tokens should be attended to by the model
        - pixel_values => Pixel values to be fed to model
        - pixel_values_videos => Pixel values of videos to be fed to model
        - image_grid_thw => List of image 3D grid in LLM
        - video_grid_thw => List of image 3D grid in LLM
        """
        output_kwargs = self._merge_kwargs(
            Qwen2VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            images_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = images_inputs["image_grid_thw"]
        else:
            images_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.image_processor(images=None, videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]
        
        # Prepare image tokens for tokenizer
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while "<|image_pad|>" in text[i]:
                    text[i] = text[i].replace(
                        "<|image_pad|>", "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", "<|image_pad|>")

        if video_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while "<|video_pad|>" in text[i]:
                    text[i] = text[i].replace(
                        "<|video_pad|>", "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", "<|video_pad|>")
        
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        for key, value in text_inputs.items():
            text_inputs[key] = to_tensor(value, self.device)
            
        for key, value in images_inputs.items():
            images_inputs[key] = to_tensor(value, self.device)

        for key, value in videos_inputs.items():
            videos_inputs[key] = to_tensor(value, self.device)
        
        data = {**text_inputs, **images_inputs, **videos_inputs}

        return data

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
    