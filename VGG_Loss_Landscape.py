import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_Light
from models.vgg import VGG_A_Dropout
from models.vgg import VGG_A_BatchNorm  # you need to implement this network
from data.loaders import get_cifar_loader

import torchvision.utils
from pathlib import Path


# This function is used to calculate the accuracy of model classification
def get_accuracy(model, dataloader, device):
    ## --------------------
    # Add code as needed
    """
    计算模型在 dataloader 上的准确率

    :param model: 待评估的模型
    :param dataloader: 数据加载器(如 val_loader)
    :param device: 'cpu' 或 'cuda'
    :return: float,准确率(0~100)
    """
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = 100.0 * correct / total
    return accuracy
    ## --------------------
    pass


# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []

    # 外层循环：epoch
    for epoch in range(epochs_n):
        print(f"\nEpoch {epoch + 1}/{epochs_n}")

        # 创建当前epoch的进度条
        epoch_progress = tqdm(total=batches_n, desc=f"Training batches", unit="batch", leave=False)

        model.train()
        loss_list = []
        learning_curve[epoch] = 0  # 重置为0

        for batch_idx, data in enumerate(train_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)

            loss_list.append(loss.item())
            learning_curve[epoch] += loss.item()

            loss.backward()
            optimizer.step()

            # 更新进度条（每个batch更新一次）
            epoch_progress.update(1)
            epoch_progress.set_postfix(loss=f"{loss.item():.4f}")

        # 完成当前epoch的进度条
        epoch_progress.close()

        losses_list.append(loss_list)
        learning_curve[epoch] /= batches_n

        # 计算验证准确率
        val_acc = get_accuracy(model, val_loader, device)
        val_accuracy_curve[epoch] = val_acc

        # 打印epoch结果
        print(f"Epoch {epoch + 1} - Loss: {learning_curve[epoch]:.4f} - Val Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > max_val_accuracy:
            max_val_accuracy = val_acc
            max_val_accuracy_epoch = epoch
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved new best model at epoch {epoch + 1} with val acc {val_acc:.2f}%")

    return losses_list, grads


# loss landscape绘制，绘制loss文件夹下所有模型
def plot_loss_landscape(loss_dir,
                        output_image="loss_landscape.png",
                        step_size=10,
                        alpha=0.4,
                        use_log_scale=True,
                        figsize=(12, 8),
                        dpi=100,
                        title="Loss Landscape Comparison",
                        xlabel="Training Steps (grouped by iterations)",
                        ylabel="Loss Value",
                        grid_style=('--', 0.7),
                        legend_loc='upper right',
                        legend_fontsize=12,
                        title_fontsize=16,
                        label_fontsize=14,
                        colors=None):
    """
    绘制多个模型的损失景观图，每个模型显示其loss值的波动范围

    参数:
        loss_dir (str): 包含loss文件的目录路径
        output_image (str): 输出图像文件名，默认为"loss_landscape.png"
        step_size (int): 每个step包含的迭代次数，默认为10
        alpha (float): 填充区域的透明度(0-1)，默认为0.4
        use_log_scale (bool): 是否使用对数y轴，默认为True
        figsize (tuple): 图像尺寸，默认为(12, 8)
        dpi (int): 图像分辨率，默认为100
        title (str): 图表标题，默认为"Loss Landscape Comparison"
        xlabel (str): x轴标签，默认为"Training Steps (grouped by iterations)"
        ylabel (str): y轴标签，默认为"Loss Value"
        grid_style (tuple): 网格线样式(linestyle, alpha)，默认为('--', 0.7)
        legend_loc (str): 图例位置，默认为'upper right'
        legend_fontsize (int): 图例字体大小，默认为12
        title_fontsize (int): 标题字体大小，默认为16
        label_fontsize (int): 轴标签字体大小，默认为14
        colors (list): 自定义颜色列表，如果为None则使用默认颜色
    """
    # 创建图形
    plt.figure(figsize=figsize, dpi=dpi)
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.grid(True, linestyle=grid_style[0], alpha=grid_style[1])

    # 设置默认颜色
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # 遍历所有loss文件
    for idx, file_path in enumerate(Path(loss_dir).glob("*.txt")):
        model_name = file_path.stem  # 使用文件名作为模型名称

        # 读取并解析文件
        all_losses = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    # 处理每行中的空格分隔的损失值
                    losses = [float(x) for x in line.split()]
                    all_losses.extend(losses)
        except Exception as e:
            print(f"错误: 读取文件 {file_path} 时出错 - {e}")
            continue

        if not all_losses:
            print(f"警告: {file_path} 没有有效数据，跳过")
            continue

        print(f"处理模型: {model_name} ({len(all_losses)} 个loss值)")

        # 计算step数量
        num_steps = len(all_losses) // step_size

        # 准备min_curve和max_curve
        min_curve = []
        max_curve = []

        # 每step_size个loss值作为一个step，计算该step的最小和最大值
        for i in range(num_steps):
            start = i * step_size
            end = start + step_size
            step_losses = all_losses[start:end]

            min_loss = min(step_losses)
            max_loss = max(step_losses)

            min_curve.append(min_loss)
            max_curve.append(max_loss)

        # 创建step数组
        steps = np.arange(num_steps)

        # 填充最小值和最大值之间的区域
        plt.fill_between(steps, min_curve, max_curve,
                         color=colors[idx % len(colors)],
                         alpha=alpha,
                         label=model_name)

    # 添加图例和美化
    plt.legend(fontsize=legend_fontsize, loc=legend_loc)
    if use_log_scale:
        plt.yscale('log')  # 对数坐标更好展示变化
    plt.tight_layout()

    # 保存并显示图像
    plt.savefig(output_image, bbox_inches='tight')
    print(f"结果已保存至: {output_image}")


if __name__ == '__main__':

    # ## Constants (parameters) initialization
    num_workers = 4
    batch_size = 128

    # add our package dir to path
    module_path = os.path.dirname(os.getcwd())
    home_path = module_path
    figures_path = os.path.join(home_path, 'reports', 'figures')
    models_path = os.path.join(home_path, 'reports', 'models')

    # Make sure you are using the right device.
    "没有GPU,这里改用CPU"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cpu")
    print(device)


    # Initialize your data loader and
    # make sure that dataloader works
    # as expected by observing one
    # sample from it.
    train_loader = get_cifar_loader(train=True)
    val_loader = get_cifar_loader(train=False)
    for X, y in train_loader:
        ## --------------------
        # Add code as needed
        # X: 图像张量，shape: [B, 3, 32, 32]
        # y: 标签张量，shape: [B]
        print("Batch shape:", X.shape)
        print("Labels shape:", y.shape)
        print("Labels:", y.tolist())

        # 将前8张图像组成网格展示（将像素还原到 0-1）
        grid = torchvision.utils.make_grid(X[:8], nrow=4, normalize=True)
        npimg = grid.permute(1, 2, 0).numpy()


        plt.figure(figsize=(6, 3))
        plt.imshow(npimg)
        plt.title("Sample Training Images")
        plt.axis("off")
        plt.savefig("sample.png")
        ## --------------------
        break

    # Train your model
    # feel free to modify
    epo = 20
    loss_save_path = 'loss/baseline'
    grad_save_path = 'grad/baseline'

    set_random_seeds(seed_value=2020, device=device)
    # model = VGG_A()
    # model = VGG_A_Light()
    # model = VGG_A_Dropout()
    model = VGG_A_BatchNorm()
    model_name = type(model).__name__

    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loss, grads = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
    np.savetxt(os.path.join(loss_save_path, f'{model_name}.txt'), loss, fmt='%s', delimiter=' ')
    np.savetxt(os.path.join(grad_save_path, f'{model_name}.txt'), grads, fmt='%s', delimiter=' ')

    # 绘制“loss”文件夹下所有模型
    plot_loss_landscape(loss_dir="loss/baseline")
