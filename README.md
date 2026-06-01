# MA-PC-SRI

**Multi-Array Physics-Consistent Self-Refining Inversion for Uncertainty-Aware ERT Imaging**

基于多阵列融合和物理自洽迭代的ERT反演框架，实现无标签训练和不确定性量化。

目标投稿：IEEE Transactions on Geoscience and Remote Sensing (TGRS)

## 核心创新

1. **物理自洽损失函数** — 利用已知正演算子替代标签数据，不需要 (m, d) 配对训练
2. **不动点迭代解码器** — 自洽迭代收敛到满足物理约束的解
3. **多链随机初始化采样** — 自然实现不确定性量化，解决非唯一性问题
4. **FNO正演代理** — Fourier Neural Operator 替代 FEM，快速可微（快1000倍）
5. **无标签训练范式** — 消除合成-野外域差距

## 方法继承

- 多阵列数据融合（WN+WB+SC）← SRERTF-Net (Liu et al., TGRS 2025)
- 传统反演结果作为物理先验 ← SRERTF-Net
- 多路径编码器 + 密集连接 ← SRERTF-Net
- 闭环一致性约束 ← GPR闭环去噪 (Liu et al., TGRS 2024)
- 粗→精两阶段策略 ← 两阶段去噪 (Liu et al., GRSL 2024)

## 环境配置

```bash
conda create -n geophys python=3.10
conda activate geophys
pip install -r requirements.txt
```

## 快速开始

### 1. 生成训练数据

```bash
# 标准训练数据
python data/generate_field_like_data.py --n_samples 100 --arrays wenner dipole-dipole schlumberger
```

### 2. 训练 FNO 正演代理

```bash
python train/train_fno.py --array wenner --epochs 200
```

### 3. 训练自洽反演模型

```bash
python train/train_selfrefine.py --array wenner --epochs 500
```

### 4. 运行完整实验

```bash
python train/run_experiments.py --full
```

### 5. 快速验证

```bash
python train/run_experiments.py --quick
```

## 项目结构

```
SelfSup-KAN-ERT/
├── models/                           # 核心模型
│   ├── FNOForward.py                 # FNO正演代理（替代FEM）
│   ├── MultiArrayEncoder.py          # 多阵列编码器（继承SRERTF-Net）
│   ├── SelfRefineDecoder.py          # 自洽迭代解码器（核心创新）
│   ├── FourierEncoding.py            # Fourier特征编码
│   ├── PCGrad.py                     # 多任务梯度协调
│   ├── forward_solver.py             # pyGIMLi FEM正演（ ground truth）
│   └── constants.py                  # 集中常量配置
├── data/
│   ├── model_generators.py           # 合成地质模型生成器
│   ├── forward_modeling.py           # ERT正演封装
│   ├── generate_field_like_data.py   # 野外级仿真数据生成
│   └── download_public_datasets.py   # 公开数据集下载
├── pretrain/
│   ├── MaskedPredNet.py              # 掩码预测编码器（参考）
│   ├── ContrastiveLoss.py            # InfoNCE对比损失（参考）
│   └── train_pretrain.py             # 预训练脚本（参考）
├── train/
│   ├── datasets.py                   # 共享数据集类
│   ├── engine.py                     # 统一训练引擎
│   ├── train_fno.py                  # FNO代理训练
│   ├── train_selfrefine.py           # 自洽反演训练
│   └── run_experiments.py            # 论文实验运行
├── eval/
│   ├── evaluate.py                   # 评估指标（含PSNR/GMSD/UQ）
│   └── plot_residuals.py             # 可视化工具
├── tests/
│   ├── test_data.py                  # 数据测试
│   └── test_new_models.py            # 新模型测试
├── results/                          # 实验输出
│   ├── fno/                          # FNO训练结果
│   ├── selfrefine/                   # 自洽反演结果
│   └── paper/                        # 论文实验结果
├── requirements.txt
└── README.md
```

## 评估指标

| 指标 | 说明 | 来源 |
|------|------|------|
| MSE/RMSE | 像素级误差 | 标准 |
| MAE | 平均绝对误差 | 标准 |
| Correlation | Pearson相关系数 | 标准 |
| SSIM | 结构相似性 | 标准 |
| PSNR | 峰值信噪比 | 新增 |
| GMSD | 梯度幅值相似性偏差 | 新增 |
| Data Misfit | 正演残差（物理一致性） | 新增 |
| Uncertainty Map | 像素级不确定性图 | 创新 |

## 与SRERTF-Net的对比

| 维度 | SRERTF-Net (Liu et al.) | MA-PC-SRI (我们的) |
|------|------------------------|-------------------|
| 训练方式 | 有监督，需要 (m,d) 配对 | **无标签，只需 d** |
| 输出 | 1个确定性模型 | **N个多样化模型 + 不确定性图** |
| 非唯一性 | 忽略 | **核心解决目标** |
| 物理约束 | 隐式（输入含物理） | **精确正演 + 自洽性** |
| 合成-野外差距 | 存在 | **消除** |

## 引用

```bibtex
@article{ma_pc_sri,
  title={MA-PC-SRI: Multi-Array Physics-Consistent Self-Refining Inversion for Uncertainty-Aware ERT Imaging},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026}
}
```

## 许可证

仅供学术研究使用。
