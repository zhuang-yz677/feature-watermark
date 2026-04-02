import argparse
from copy import deepcopy

from htgn_network import HashToGaussianGenerator
from torchvision import transforms
import torch
import os

from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, DDIMInverseScheduler
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from utils import *
import open_clip
import json

from image_utils import *
import time
from vectors_generation import init_vectors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.set_printoptions(precision=4, sci_mode=False)


parser = argparse.ArgumentParser()
parser.add_argument('--hash_dim', default=64, type=int)
parser.add_argument('--noise_dim', default=16384, type=int)
parser.add_argument('--batch_s', default=1000, type=int, help='the number of watermarks')
parser.add_argument('--threshold', default=0.5, type=int, help='threshold for distinguishing watermark matching')
parser.add_argument('--reference_model', default='ViT-g-14', type=str, help='CLIP model')
parser.add_argument('--reference_model_pretrain', default='laion2b_s12b_b42k', type=str, help='CLIP model')

args = parser.parse_args()

hash_dim = args.hash_dim
noise_dim = args.noise_dim
batch_s = args.batch_s
threshold = args.threshold

model_path = 'models/'
model_name = 'best_model/'
real_model = 'best'
# 水印特征向量
vector_path = f'./{model_path}/pretrain_models/vectors_{hash_dim}_{batch_s}.pth'
# 图像反演噪声
test_path = f'./{model_path}/pretrain_models/inv_noise_{hash_dim}_{batch_s}_best.pth'

# 加载模型
print(f'using model: {model_name + real_model}')
test_model = HashToGaussianGenerator(hash_dim, noise_dim).to(device)
checkpoint = torch.load(model_path + model_name + real_model + '.pth')
test_model.load_state_dict(checkpoint)

test_model.eval()

# 加载水印
if not os.path.exists(vector_path):
    vectors = init_vectors(hash_dim, batch_s, save=True)
else:
    vectors = torch.load(vector_path)
dataset = TensorDataset(vectors.to(device))
dataloader = DataLoader(dataset, batch_size=batch_s, drop_last=True)


with torch.no_grad():
    index, p = 0, 0
    generated = []
    hash_tuple = []
    for i, image in enumerate(dataloader):
        test_hash = image[0].to(device)

        # size_in_bytes = test_hash.element_size() * test_hash.nelement()
        #
        # size_in_mb = size_in_bytes / (1024 ** 2)
        #
        # print(f"占用内存: {size_in_bytes} Bytes")
        # print(f"占用内存: {size_in_mb:.2f} MB")
        # print(f"平均：{size_in_bytes / len(test_hash)}")

        # watermark -> noise
        generated_noise, _ = test_model(test_hash)

        # noise -> watermark
        _, inversed_hash = test_model(generated_noise, rev=True)
        inversed_hash.data = torch.nn.functional.normalize(inversed_hash.data)

    hash_tuple.append([test_hash, inversed_hash])

# print(f'using num.{index} max_p:{p}')

ori_hash = hash_tuple[index][0].unsqueeze(1)
dec_hash = hash_tuple[index][1].unsqueeze(0)

# # 计算原始hash和解码hash的余弦相似度
# cos_sim = torch.cosine_similarity(ori_hash, dec_hash, dim=-1)
#
# print(cos_sim)
# print(torch.mean(cos_sim).item())
#
# # 生成对角线掩码
# mask = torch.eye(batch_s, dtype=torch.bool)
# non_diag_mask = ~mask
# # 上三角矩阵
# upper_mask = torch.triu(torch.ones_like(cos_sim), diagonal=1).bool()
#
# # 通过统计对角线元素与每行最大元素，寻找匹配成功的水印
# max_col_indices = torch.argmax(cos_sim, dim=1)  # [n]
# diag_col_indices = torch.arange(batch_s).to(device)  # [n]，值为[0,1,2,...,n-1]
# diag_values = torch.diag(cos_sim)
#
# correct_count = torch.sum((max_col_indices == diag_col_indices) & (diag_values > threshold)).item()
# # 计算正确率
# correct_ratio = correct_count / batch_s
# print(correct_ratio)
#
# avg = round(torch.mean(cos_sim[mask]).item(), 4)
# count0 = torch.sum((cos_sim[mask] > 0.95)).item() / batch_s
#
# print(f'平均余弦相似度：{avg}')
# print(f'正确率：{correct_ratio}')


if not os.path.exists(test_path):
    # 加载clip模型
    ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model,
                                                                              pretrained=args.reference_model_pretrain,
                                                                              device=device)
    ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # 加载diffusion模型
    model_id = 'stabilityai/stable-diffusion-2-1-base'
    # model_id = '~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1-base/snapshots/1f758383196d38df1dfe523ddb1030f2bfab7741/'

    data_id = 'Gustavosta/Stable-Diffusion-Prompts'
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder='scheduler', local_files_only=True)
    inverse_scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder='scheduler', local_files_only=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        local_files_only=True
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    # sd提示词
    prompt_dataset = load_dataset(data_id)['train']
    prompt_key = 'Prompt'

    # # coco提示词
    # with open('/home/hdda/zyz/HashtoNoise/data/fid_outputs/coco/meta_data.json') as f:
    #     dataset = json.load(f)
    #     image_files = dataset['images']
    #     dataset = dataset['annotations']
    #     prompt_key = 'caption'

    temp = 0
    init_noise = torch.reshape(generated[index], (len(generated_noise), 4, 64, 64))

    image_list = []
    clip_scores = 0
    for i, noise in enumerate(init_noise):
        print(f'generating... {i} / {len(init_noise)}')

        # 获取提示词
        p_index = random.randint(0, len(prompt_dataset) - 1)
        current_prompt = prompt_dataset[p_index][prompt_key]

        # current_prompt = dataset[i][prompt_key]

        noise = noise.unsqueeze(0).half()

        output = pipe(
            current_prompt,
            num_images_per_prompt=1,
            guidance_scale=7.5,
            num_inference_steps=50,
            height=512,
            width=512,
            latents=noise,
        ).images[0]

        tmp_clip = measure_similarity(output, current_prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device)
        clip_scores += tmp_clip
        # print(tmp_clip)

        # name = image_files[i]['file_name']
        # output.save(f'./data/fid_outputs/coco/fid_test/{name}')
        image_list.append(output)

    print(f'clip_gen:{clip_scores / batch_s}')

    # with open(f'./results/result_{hash_dim}_{real_model}.txt', 'w+') as f:
    #     f.write(f'clip_gen:{clip_value / batch_s}\n')
    #     # f.write(f'fid_gen:{fid_gen}')# fid_ref:{fid_ref}\n')

    # 设置为反演状态
    pipe.scheduler = inverse_scheduler
    ldm_encoder = deepcopy(pipe.vae).float()

    ldm_encoder.eval()
    ldm_encoder.to(device)

    distorted_image_list = []

    for i, image in enumerate(image_list):
        # 对图像添加扰动
        distorted_image_list.append([])
        for item in distortion_options:
            method, img = image_distortion(image, **item)
            distorted_image_list[-1].append([method, img])
            name = [f"{key}_{value}" for key, value in method.items()][0]

    hash_list = []
    noise_list = []
    with torch.no_grad():
        # 图像反演
        for i, item in enumerate(distorted_image_list):
            print(f'inversing... {i} / {len(distorted_image_list)}')
            temp_hash_list = []
            temp_noise_list = []
            for e, (m, image_d) in enumerate(item):
                # Image转tensor
                image_tensor = transform_img(image_d).unsqueeze(0).to(ldm_encoder.dtype).to(device)

                image_latents_w = ldm_encoder.encode(image_tensor).latent_dist.mode().to(pipe.unet.dtype) * 0.18215
                reversed_noise = pipe(
                    "",
                    latents=image_latents_w,
                    guidance_scale=1,
                    num_inference_steps=50,
                    output_type='latent'
                ).images.flatten(1).to(dtype=torch.float)
                temp_noise_list.append(reversed_noise)

            noise_list.append(torch.stack(temp_noise_list))

        all_noise_tensor = torch.stack(noise_list) # [n,22,1,16384]
        torch.save(all_noise_tensor, test_path)

print("-"*20)
all_noise_tensor = torch.load(test_path) # [n,22,1,16384]

for i in range(len(distortion_options)):
    method = distortion_options[i]
    reversed_noise = all_noise_tensor[:, i, :, :].squeeze(1)

    _, inv_hash = test_model(reversed_noise, rev=True)
    inv_hash.data = torch.nn.functional.normalize(inv_hash.data)

    hash_tensors = 0
    # 计算原始水印与重建水印的相似度
    inverse_cos = torch.cosine_similarity(ori_hash, inv_hash, dim=-1)

    # 通过统计对角线元素与每行最大元素，寻找匹配成功的水印
    max_col_indices = torch.argmax(inverse_cos, dim=1)  # [n]
    diag_col_indices = torch.arange(batch_s).to(device)  # [n]，值为[0,1,2,...,n-1]
    diag_values = torch.diag(inverse_cos)
    correct_count = torch.sum((max_col_indices == diag_col_indices) & (diag_values > threshold)).item()
    # 计算正确率
    correct_ratio = correct_count / batch_s
    avg = round(torch.mean(diag_values).item(), 4)

    print(f'扰动方式：{method}')
    print(f'平均余弦相似度：{avg}')
    print(f'正确率：{correct_ratio}')

    with open(f'./results/result_{hash_dim}_{real_model}.txt', 'a+') as f:
        s = f'method：{method} AVG：{avg} accuracy：{correct_ratio}\n'
        f.write(s)

