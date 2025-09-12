import torch, matplotlib.pyplot as plt

def plot_sparsity(tensor, save="runs/sparsity.png"):
    # tensor: 任意层的 sINT 激活 [B,T,D]
    with torch.no_grad():
        act = (tensor != 0).float().mean(dim=(0,2)).cpu()  # 每个时间步的激活比例
    plt.figure(); plt.plot(act.numpy()); plt.xlabel("t"); plt.ylabel("active ratio")
    plt.savefig(save); plt.close()
