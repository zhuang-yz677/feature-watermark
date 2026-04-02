import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from scipy import stats
import matplotlib.pyplot as plt
from PIL import Image
import random

# 判断是否为标准高斯噪声
def is_gaussian_noise(tensor: torch.Tensor):

    # 转换为numpy数组
    arr = tensor.cpu().detach().numpy().astype(np.float64).flatten()

    # 均值+方差+ks检验
    mean = np.mean(arr)
    std = np.std(arr)
    _, p_value = stats.kstest(arr, 'norm', args=(0, 1))

    return mean.item(), std.item(), p_value.item()


def draw(generated_noise, model_name, mode, epoch):
    plt.figure(figsize=(10, 6))

    # 绘制生成噪声
    plt.hist(generated_noise.cpu().detach().numpy().flatten(), bins=200, density=True, alpha=0.6, label='Generated')

    # 绘制标准高斯分布曲线
    x = np.linspace(-10, 10, 2000)
    plt.plot(x, stats.norm.pdf(x, 0, 1), 'r-', label='Standard Normal')
    plt.legend()
    plt.title('Distribution of Generated Noise')

    if not os.path.exists(f'./images/{model_name}/{mode}/'):
        os.makedirs(f'./images/{model_name}/{mode}/')
    plt.savefig(f'./images/{model_name}/{mode}/{epoch}.png')

    plt.close()


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def measure_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = clip_preprocess(images).unsqueeze(0).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return (image_features @ text_features.T).mean(-1)