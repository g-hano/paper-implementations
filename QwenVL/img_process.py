import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from transformers.image_utils import PILImageResampling
from PIL import Image
import torch
from transformers.image_transforms import to_channel_dimension_format, resize

OPENAI_CLIP_MEAN = [
    0.48145466,
    0.4578275,
    0.40821073
]
OPENAI_CLIP_STD = [
    0.26862954,
    0.26130258,
    0.27577711
]

def convert_to_rgb(image):
    return image.convert("RGB")

def infer_channel_dimension_format(image, num_channels=None):
    num_channels = num_channels if num_channels is not None else (1, 3)
    num_channels = (num_channels,) if isinstance(num_channels, int) else num_channels

    if image.ndim == 3:
        first_dim, last_dim = 0, 2
    elif image.ndim == 4:
        first_dim, last_dim = 1, 3

    if image.shape[first_dim] in num_channels:
        return "channels_first"
    elif image.shape[last_dim] in num_channels:
        return "channels_last"
    
def rescale(
    image: np.ndarray,
    scale: float,
    data_format=None,
    dtype: np.dtype = np.float32,
    input_data_format=None,
) -> np.ndarray:

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input image must be of type np.ndarray, got {type(image)}")
    
    rescaled_image = image * scale
    if data_format is not None:
        rescaled_image = to_channel_dimension_format(rescaled_image, data_format, input_data_format)

    rescaled_image = rescaled_image.astype(dtype)

    return rescaled_image
    
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, 
    max_pixels: int = 14 * 14 * 4 * 1280
):
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar

def get_image_size(image, channel_dim):
    if isinstance(image, Image.Image):  # Check if image is a PIL Image
        return image.size
    if channel_dim.lower() == "channels_first":
        return image.shape[-2], image.shape[-1]
    elif channel_dim.lower() == "channels_last":
        return image.shape[-3], image.shape[-2]

def normalize(image, mean, std, data_format=None, input_data_format=None):
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    
    if input_data_format == "channels_first":
        ch_dim_axis = image.ndim - 3
    else:
        ch_dim_axis = image.ndim - 1

    num_channels = image.shape[ch_dim_axis]

    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)
    
    if isinstance(mean, (list, tuple, np.ndarray)):
        if len(mean) != num_channels:
            raise ValueError(f"mean must have {num_channels} elements if it is an iterable, got {len(mean)}")
    else:
        mean = [mean] * num_channels
    mean = np.array(mean, dtype=image.dtype)

    # Handle std
    if isinstance(std, (list, tuple, np.ndarray)):
        if len(std) != num_channels:
            raise ValueError(f"std must have {num_channels} elements if it is an iterable, got {len(std)}")
    else:
        std = [std] * num_channels
    std = np.array(std, dtype=image.dtype)
    
    if input_data_format == "channels_last":
        mean = mean.reshape((1, 1, -1))
        std = std.reshape((1, 1, -1))
    else:
        mean = mean.reshape((-1, 1, 1))
        std = std.reshape((-1, 1, 1))

    # Perform normalization
    image = (image - mean) / std

    image = to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
    return image

class Qwen2VLImageProcessor:
    model_input_names = [
        "pixel_values", 
        "image_grid_thw", 
        "pixel_values_videos", 
        "video_grid_thw"
    ]

    def __init__(
        self,
        do_resize = True,
        resample = PILImageResampling.BICUBIC,
        do_rescale = True,
        rescale_factor = 1 / 255,
        do_normalize = True,
        image_mean = None,
        image_std = None,
        do_convert_rgb = True,
        min_pixels: int = 3136,
        max_pixels: int = 12845056,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2
    ):
        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.size = {"min_pixels": min_pixels, "max_pixels": max_pixels}
        self.do_convert_rgb = do_convert_rgb

    def _preprocess(
        self,
        images,
        do_resize = None,
        resample = None,
        do_rescale = None,
        rescale_factor = None,
        do_normalize = None,
        image_mean = None,
        image_std = None,
        do_convert_rgb = None,
        data_format = "channels_first",
        input_data_format = None
    ):
        if not isinstance(images, list):
            images = [images]

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]      

        images = [np.array(image) for image in images]
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])    

        h, w = get_image_size(images[0], input_data_format)

        resized_h, resized_w = h, w
        processed_images = []

        for image in images:
            if do_resize:
                resized_h, resized_w = smart_resize(
                    height=h,
                    width=w,
                    factor=self.patch_size * self.merge_size,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels
                )

                image = resize(
                    image, size=(resized_h, resized_w), resample=resample
                )

            if do_rescale:
                image = rescale(image, 
                                scale=rescale_factor, 
                                input_data_format=input_data_format)

            if do_normalize:
                image = normalize(image, mean=image_mean, std=image_std, input_data_format=input_data_format)

            image = to_channel_dimension_format(image, data_format)
            processed_images.append(image)
        
        patches = np.array(processed_images)
        if data_format.lower() == "channels_last":
            patches = patches.transpose(0, 3, 1, 2)
        if patches.shape[0] == 1:
            patches = np.tile(patches, (self.temporal_patch_size, 1, 1, 1))
        
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h = resized_h // self.patch_size
        grid_w = resized_w // self.patch_size

        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
        )
        print(f"{flatten_patches.shape=} {flatten_patches.dtype=}")
        print(f"{grid_t=} {grid_h=} {grid_w=}")
        print(f"{type(grid_t)=} {type(grid_h)=} {type(grid_w)=}")
        return flatten_patches, (grid_t, grid_h, grid_w)
    
    def preprocess(
        self,
        images,
        videos = None,
        do_resize = None,
        size = None,
        resample = None,
        do_rescale = None,
        rescale_factor = None,
        do_normalize = None,
        image_mean = None,
        image_std = None,
        do_convert_rgb = None,
        data_format = "channels_first",
        input_data_format = None
    ):
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        if images is not None:
            if not isinstance(images, list):
                images = [images]
        if videos is not None:
            if not isinstance(videos, list):
                videos = [videos]

        if images is not None:
            pixel_values = []
            vision_grid_thws = []
            for image in images:
                patches, image_grid_thw = self._preprocess(
                    image,
                    do_resize=do_resize,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(image_grid_thw)
            pixel_values = np.array(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)
            data = {
                "pixel_values": torch.tensor(pixel_values), 
                "image_grid_thw": torch.tensor(vision_grid_thws)
            }
        
        if videos is not None:
            pixel_values = []
            vision_grid_thws = []
            for video in videos:
                patches, video_grid_thw = self._preprocess(
                    video,
                    do_resize=do_resize,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(video_grid_thw)
            pixel_values = np.array(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)
            data = {
                "pixel_values_videos": pixel_values, 
                "video_grid_thw": vision_grid_thws
            }
        
        return data

    def __call__(self, images, **kwargs):
        return self.preprocess(images, **kwargs)