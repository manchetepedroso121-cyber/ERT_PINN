"""
PCGrad - Project Conflicting Gradients
梯度手术：解决多任务学习中的梯度冲突问题

参考: Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020

在PINN中，数据损失和PDE损失的梯度可能冲突：
- 数据损失推动网络拟合观测数据
- PDE损失推动网络满足物理方程
PCGrad检测并消除冲突梯度分量，使两个任务协调优化
"""

import torch
import torch.nn as nn
import numpy as np


class PCGrad:
    """PCGrad优化器包装器

    在每次 step() 中：
    1. 计算每个任务的梯度
    2. 检测梯度之间的余弦相似度
    3. 如果 < 0（冲突），投影去除冲突分量
    4. 使用修正后的梯度更新参数

    Args:
        optimizer: 基础优化器 (如 Adam, AdamW)
        reduction: 梯度聚合方式 ('mean' 或 'sum')
    """

    def __init__(self, optimizer, reduction='mean'):
        self.optimizer = optimizer
        self.reduction = reduction

    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()

    def pc_backward(self, objectives):
        """计算PCGrad修正后的梯度

        Args:
            objectives: 损失值列表 [loss1, loss2, ...]

        Raises:
            ValueError: if objectives is empty
        """
        if not objectives:
            raise ValueError("objectives list is empty")
        # 收集每个任务的梯度（只收集有梯度的参数）
        param_grads = []  # list of {param_idx: grad}
        all_params = []   # 有序参数列表

        for group in self.optimizer.param_groups:
            for p in group['params']:
                all_params.append(p)

        for obj in objectives:
            self.optimizer.zero_grad()
            obj.backward(retain_graph=True)
            grads = {}
            for idx, p in enumerate(all_params):
                if p.grad is not None:
                    grads[idx] = p.grad.clone()
            param_grads.append(grads)

        # PCGrad: 投影去除冲突梯度
        pc_grads = self._project_conflicting(param_grads, len(all_params))

        # 将修正后的梯度写回参数
        for idx, p in enumerate(all_params):
            if idx in pc_grads:
                if p.grad is None:
                    p.grad = pc_grads[idx]
                else:
                    p.grad.copy_(pc_grads[idx])

    def _project_conflicting(self, param_grads, n_params):
        """投影去除冲突梯度

        对每对梯度 (g_i, g_j)：
        - 如果 g_i · g_j < 0（冲突），将 g_i 投影到 g_j 的正交补空间
        - 否则保持不变

        Args:
            param_grads: list of {param_idx: grad_tensor} 每个任务的梯度字典
            n_params: 参数总数

        Returns:
            pc_grads: {param_idx: grad_tensor} 修正后的梯度
        """
        n_tasks = len(param_grads)
        pc_grads = {}

        for i in range(n_params):
            # 收集所有任务在第i个参数上的梯度
            g_list = []
            task_indices = []
            for t in range(n_tasks):
                if i in param_grads[t]:
                    g_list.append(param_grads[t][i].reshape(-1))
                    task_indices.append(t)

            if len(g_list) == 0:
                continue

            if len(g_list) == 1:
                pc_grads[i] = g_list[0].reshape(param_grads[task_indices[0]][i].shape)
                continue

            # 对每个任务的梯度进行投影
            projected = []
            for t_idx in range(len(g_list)):
                g = g_list[t_idx].clone()
                # 随机打乱其他任务的顺序（论文建议）
                perm = torch.randperm(len(g_list))
                for j_idx in perm:
                    if j_idx == t_idx:
                        continue
                    # 计算余弦相似度
                    dot = torch.dot(g, g_list[j_idx])
                    if dot < 0:  # 冲突
                        # 投影: g = g - (g·g_j / ||g_j||^2) * g_j
                        g = g - (dot / (g_list[j_idx].norm() ** 2 + 1e-8)) * g_list[j_idx]
                projected.append(g)

            # 聚合
            stacked = torch.stack(projected)
            if self.reduction == 'mean':
                pc_grads[i] = stacked.mean(dim=0).reshape(param_grads[task_indices[0]][i].shape)
            else:
                pc_grads[i] = stacked.sum(dim=0).reshape(param_grads[task_indices[0]][i].shape)

        return pc_grads

    def step(self):
        """执行优化步骤"""
        self.optimizer.step()


if __name__ == '__main__':
    print("=" * 50)
    print("Test 1: PCGrad basic")
    model = nn.Linear(4, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    pc_optimizer = PCGrad(optimizer)

    x = torch.randn(8, 4)
    target1 = torch.randn(8, 1)
    target2 = torch.randn(8, 1)

    pred = model(x)
    loss1 = nn.MSELoss()(pred, target1)
    loss2 = nn.MSELoss()(pred, target2)

    pc_optimizer.zero_grad()
    pc_optimizer.pc_backward([loss1, loss2])
    pc_optimizer.step()

    print(f"  Loss1: {loss1.item():.4f}")
    print(f"  Loss2: {loss2.item():.4f}")
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(f"  {name} grad norm: {p.grad.norm().item():.6f}")

    print("\n" + "=" * 50)
    print("Test 2: PCGrad conflicting gradients")
    model2 = nn.Linear(2, 1, bias=False)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.1)
    pc2 = PCGrad(opt2)

    x = torch.tensor([[1.0, 0.0]])
    pred = model2(x)
    loss1 = pred      # 推动权重变大
    loss2 = -pred     # 推动权重变小（冲突）

    pc2.zero_grad()
    pc2.pc_backward([loss1, loss2])
    grad_after = model2.weight.grad.clone()
    print(f"  Conflicting grad norm after PCGrad: {grad_after.norm().item():.6f}")
    print(f"  Should be near zero: {grad_after.norm().item() < 0.1}")

    print("\n" + "=" * 50)
    print("Test 3: PCGrad with pykan-style unused params")
    # 模拟有参数不参与计算图的情况
    model3 = nn.Linear(4, 1)
    extra_param = nn.Parameter(torch.zeros(2))  # 不参与loss计算的参数
    opt3 = torch.optim.Adam([{'params': model3.parameters()},
                              {'params': [extra_param]}], lr=0.01)
    pc3 = PCGrad(opt3)

    pred = model3(torch.randn(4, 4))
    loss1 = pred.sum()
    loss2 = (pred ** 2).sum()

    pc3.zero_grad()
    pc3.pc_backward([loss1, loss2])
    pc3.step()
    print(f"  extra_param grad: {extra_param.grad}")
    print(f"  model params grad OK: {model3.weight.grad is not None}")
