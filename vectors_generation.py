import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_vectors(dim, num_vectors, epsilon=0.2, save=False):
    # 随机初始化向量并归一化
    vectors = torch.rand(num_vectors, dim, requires_grad=True)#.to(device)
    vectors.data = torch.nn.functional.normalize(vectors.data, dim=1)

    # 优化器
    optimizer = torch.optim.Adam([vectors], lr=0.01)

    epochs = 10000
    loss_history, max_history = [], []

    for epoch in range(epochs):
        # 内积矩阵
        inner_products = torch.matmul(vectors, vectors.t())

        # 上三角矩阵
        mask = torch.triu(torch.ones_like(inner_products), diagonal=1).bool()
        off_diag_products = inner_products[mask]

        # 对于|x| > epsilon的部分，损失为(|x| - epsilon)^2，否则为0
        loss = torch.mean(torch.clamp(torch.abs(off_diag_products) - epsilon, min=0) ** 2)
        m = torch.max(off_diag_products)

        if epoch >= 2 and loss.item() > loss_history[-1] and m > max_history[-1]:
            print(f'break in epoch {epoch}')
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        max_history.append(m.item())
        loss_history.append(loss.item())

        # 重新归一化
        with torch.no_grad():
            vectors.data = torch.nn.functional.normalize(vectors.data, dim=1)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}, Max: {m.item():.6f}")

    if save:
        torch.save(vectors, f'./models/pretrain_models/vectors_{dim}_{num_vectors}.pth')

    return vectors

if __name__ == "__main__":
    # 向量参数
    dim = 64
    num_vectors = 10000
    epsilon = 0.2  # 近似正交阈值

    vectors = init_vectors(dim, num_vectors, epsilon)
