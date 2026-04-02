import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import torch.optim as optim
from utils import *
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from htgn_network import HashToGaussianGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
current_time = time.time()


def kl_divergence_loss(mean, std):
    # 计算KL散度
    kl = 0.5 * (mean**2 + std**2 - 1 - 2 * torch.log(std))

    return kl

def enhanced_loss(generated_noise):
    total_loss = 0
    # 不计算batch，而是计算单噪声损失
    for noise in generated_noise:
        # 基础KL散度损失
        mu = torch.mean(noise)
        logvar = torch.std(noise)
        kl = kl_divergence_loss(mu, logvar)

        total_loss += kl

    return total_loss


def train_generator(model, fid_model, num_samples, hash_dim, noise_dim, epochs, batch_size, lr, use_amp=True):

    model_name = 'dim' + str(hash_dim) + '+batchsize' + str(batch_size) + '/'
    model_path = 'new_model/' + model_name
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model = model.to(device)

    vectors = torch.load(f'./models/pretrain_models/vectors_{hash_dim}_{num_samples}.pth')
    dataset = TensorDataset(vectors.to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"dataset size: {len(dataset)}")

    # 定义优化器
    optimizer = optim.Adam(model.encoder.parameters(), lr=lr)
    # 混合精度训练设置
    scaler = GradScaler(enabled=use_amp)
    _p = 0
    stage = 20 # 一阶段epoch数量
    criterion = nn.MSELoss()

    # 训练
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # 冻结解码器，训练生成器
        if epoch < stage:

            for param in model.decoder.parameters():
                param.requires_grad = False

            # 加载数据
            for i, tmp in enumerate(dataloader):
                optimizer.zero_grad()
                batch_hash = tmp[0].to(device)

                # 混合精度前向传播
                with autocast(enabled=use_amp):
                    generated_noise, _ = model(batch_hash)

                kl = enhanced_loss(generated_noise)

                loss = kl / batch_size
                # 混合精度反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # print(loss.item())
                running_loss += loss.item()

        # 冻结生成器，训练解码器
        else:

            for p in model.decoder.parameters():
                p.requires_grad = True
            for p in model.encoder.parameters():
                p.requires_grad = False

            model.encoder.eval()
            optimizer = optim.Adam(model.decoder.parameters(), lr=0.01)

            torch.manual_seed(42)
            # trans = transforms.ToTensor()

            # 加载数据
            for batch_num, tmp in enumerate(dataloader):
                optimizer.zero_grad()
                batch_hash = tmp[0].to(device)

                with torch.no_grad():
                    generated_noise, _ = model(batch_hash)


                with autocast(enabled=use_amp):
                    correct_noise_clean, inversed_hash_clean = model(generated_noise, rev=True)
                inversed_hash_clean.data = torch.nn.functional.normalize(inversed_hash_clean.data)
                # inversed_hash_distorted.data = torch.nn.functional.normalize(inversed_hash_distorted.data)

                # MSE损失
                loss = criterion(batch_hash, inversed_hash_clean)

                # 混合精度反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

        # writer.add_scalar("Loss/train", running_loss, epoch)
        # for name, param in model.named_parameters():
        #     writer.add_histogram(f'{name}/weight', param, epoch)

        # 保存日志
        avg_loss = running_loss#  / len(dataloader)
        with open('./log/{}.txt'.format(current_time), 'a+') as f:
            s = f'Epoch [{epoch}/{epochs}], Loss: {avg_loss:.6f}\n'
            print(s, end='')
            f.write(s)

        # 验证效果
        model.eval()
        with torch.no_grad():
            # temp = init_vectors(hash_dim, 1000).data.to(device)
            n = 1000
            temp = torch.load(f'./models/pretrain_models/vectors_{hash_dim}_{n}.pth').to(device)
            generated_noise, inversed_hash = model(temp)
            inversed_hash.data = torch.nn.functional.normalize(inversed_hash.data)
            if epoch <= stage:
                # mean, std, p_value
                m, s, p = 0, 0, 0
                low = 0
                for item in generated_noise:
                    _m, _s, _p = is_gaussian_noise(item)
                    if _p < 0.05:
                        low += 1
                    m += _m
                    s += _s
                    p += _p
                print(f'noise:', [m / n, s / n, p / n], f'low: {low / n}')
            cos_sim = torch.cosine_similarity(temp, inversed_hash, dim=-1)
            print(cos_sim.mean(), torch.where(cos_sim < 0.5)[0].numel())

        # draw(generated_noise, model_name, epoch)
        if (epoch+1) % 5 == 0 and torch.where(cos_sim < 0.5)[0].numel() == 0 or epoch == stage-1:
            torch.save(model.state_dict(), model_path + f'epoch_{epoch}.pth')

    return model


if __name__ == "__main__":
    # 数据
    num_samples = 10000
    hash_dim = 64
    noise_dim = 16384
    epochs = 100
    batch_size = 512
    learning_rate = 0.0001

    # 初始化模型
    model = HashToGaussianGenerator(hash_dim, noise_dim).to(device)

    # 训练模型
    trained_model = train_generator(
        model,
        fid_model=None,
        num_samples=num_samples,
        hash_dim=hash_dim,
        noise_dim=noise_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
        use_amp=False
    )