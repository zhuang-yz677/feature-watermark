import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFilter
import copy


distortion_options = [
    {'none': True},
    {'jpeg_ratio': 75},
    {'jpeg_ratio': 50},
    {'jpeg_ratio': 25},
    {'random_crop_ratio': 0.9},
    {'random_crop_ratio': 0.7},
    {'random_crop_ratio': 0.5},
    {'random_drop_ratio': 0.25},
    {'random_drop_ratio': 0.5},
    {'random_drop_ratio': 0.75},
    {'resize_ratio': 0.75},
    {'resize_ratio': 0.5},
    {'resize_ratio': 0.25},
    {'gaussian_blur_r': 2},
    {'gaussian_blur_r': 4},
    {'gaussian_blur_r': 6},
    {'gaussian_std': 0.05},
    {'gaussian_std': 0.1},
    {'gaussian_std': 0.15},
    {'brightness_factor': 2},
    {'brightness_factor': 4},
    {'brightness_factor': 6}
]


def image_distortion(img,
                     none=None,
                     jpeg_ratio=None,
                     rotate_degree=None,
                     random_crop_ratio=None,
                     random_drop_ratio=None,
                     resize_ratio=None,
                     gaussian_blur_r=None,
                     median_blur_k=None,
                     gaussian_std=None,
                     sp_prob=None,
                     brightness_factor=None):

    if none:
        method = {'none': True}
        img = img

    if jpeg_ratio is not None:
        method = {'jpeg_ratio': jpeg_ratio}
        img.save(f"tmp_{jpeg_ratio}.jpg", quality=jpeg_ratio)
        with open(f"tmp_{jpeg_ratio}.jpg", 'rb') as fp:
            img = copy.copy(Image.open(fp))

    if rotate_degree is not None:
        method = {'rotate_degree': rotate_degree}
        img = transforms.RandomRotation((rotate_degree, rotate_degree))(img)

    if random_crop_ratio is not None:
        method = {'random_crop_ratio': random_crop_ratio}
        width, height, c = np.array(img).shape
        img = np.array(img)
        new_width = int(width * random_crop_ratio)
        new_height = int(height * random_crop_ratio)
        start_x = np.random.randint(0, width - new_width + 1)
        start_y = np.random.randint(0, height - new_height + 1)
        end_x = start_x + new_width
        end_y = start_y + new_height
        padded_image = np.zeros_like(img)
        padded_image[start_y:end_y, start_x:end_x] = img[start_y:end_y, start_x:end_x]
        img = Image.fromarray(padded_image)

        # img = transforms.RandomResizedCrop(img.size, scale=(random_crop_ratio, random_crop_ratio), ratio=(random_crop_ratio, random_crop_ratio))(img)

    if random_drop_ratio is not None:
        method = {'random_drop_ratio': random_drop_ratio}
        width, height, c = np.array(img).shape
        img = np.array(img)
        new_width = int(width * random_drop_ratio)
        new_height = int(height * random_drop_ratio)
        start_x = np.random.randint(0, width - new_width + 1)
        start_y = np.random.randint(0, height - new_height + 1)
        padded_image = np.zeros_like(img[start_y:start_y + new_height, start_x:start_x + new_width])
        img[start_y:start_y + new_height, start_x:start_x + new_width] = padded_image
        img = Image.fromarray(img)

    if resize_ratio is not None:
        method = {'resize_ratio': resize_ratio}
        img_shape = np.array(img).shape
        resize_size = int(img_shape[0] * resize_ratio)
        img = transforms.Resize(size=resize_size)(img)
        img = transforms.Resize(size=img_shape[0])(img)

    if gaussian_blur_r is not None:
        method = {'gaussian_blur_r': gaussian_blur_r}
        img = img.filter(ImageFilter.GaussianBlur(radius=gaussian_blur_r))

    if median_blur_k is not None:
        method = {'median_blur_k': median_blur_k}
        img = img.filter(ImageFilter.MedianFilter(median_blur_k))


    if gaussian_std is not None:
        method = {'gaussian_std': gaussian_std}
        img_shape = np.array(img).shape
        g_noise = np.random.normal(0, gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img = Image.fromarray(np.clip(np.array(img) + g_noise, 0, 255))

    if sp_prob is not None:
        method = {'sp_prob': sp_prob}
        c,h,w = np.array(img).shape
        prob_zero = sp_prob / 2
        prob_one = 1 - prob_zero
        rdn = np.random.rand(c,h,w)
        img = np.where(rdn > prob_one, np.zeros_like(img), img)
        img = np.where(rdn < prob_zero, np.ones_like(img)*255, img)
        img = Image.fromarray(img)

    if brightness_factor is not None:
        method = {'brightness_factor': brightness_factor}
        img = transforms.ColorJitter(brightness=brightness_factor)(img)

    return method, img
