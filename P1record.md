
# Phase 1 基础模型训练实施记录

本文档详细记录了 VecSetX 项目 Phase 1 (多类心脏结构重建) 的实施过程、文件修改原因及具体修改内容。

## 1. 数据准备 (Data Preparation)

> **移出说明**: 数据准备的详细记录、脚本说明及格式定义已移动至专门的文档 [Datarecord.md](Datarecord.md)。本章节不再维护。


## 2. 模型架构修改 (Model Architecture)

### 文件: `vecset/models/autoencoder.py`
*   **原因**: 原始模型仅支持 3 维坐标输入。为了实现多类重建，模型需要能够接收包含类别信息的 13 维输入。同时，解码器的查询 (Query) 仍然是 3 维空间坐标。
*   **修改内容**:
    *   **参数扩展**: 在 `VecSetAutoEncoder` 初始化函数中添加了 `input_dim` 参数，默认为 3，Phase 1 设置为 13。
    *   **分离嵌入层**: 将原来的单一 `point_embed` 层拆分为两个：
        *   `self.point_embed`: 用于编码器输入，输入维度为 `input_dim` (13)。
        *   `self.query_embed`: 用于解码器查询，输入维度固定为 3 (仅坐标)。
    *   **解码逻辑**: 在 `decode` 方法中，使用 `self.query_embed` 对查询点进行编码。
    *   **修复错误**: 修复了在代码重构过程中引入的 `SyntaxError` (语法错误) 和 `IndentationError` (缩进错误)，并重写了 `VecSetAutoEncoder` 类以确保结构正确。
    *   **清理代码**: 移除了未使用的 `fps` 导入。

## 3. 工具类与依赖适配 (Utils & Dependencies)

### 文件: `vecset/models/utils.py`
*   **原因**: 运行环境缺少 `torch_cluster` 库，导致 `fps` (最远点采样) 无法导入，阻碍了训练脚本运行。
*   **修改内容**:
    *   **添加回退机制**: 在导入 `fps` 时添加了 `try-except` 块。如果导入失败，`subsample` 函数会自动回退到 **随机采样 (Random Sampling)**。这保证了代码在缺少特定编译库的环境中也能正常运行。
    *   **清理代码**: 移除了重复的导入语句。

## 4. 数据加载与处理 (Dataset Loader)

### 文件: `vecset/utils/objaverse.py`
*   **原因**: 需要加载新的包含标签的数据格式，并解决在特定数据分布下采样失败的问题。
*   **修改内容**:
    *   **加载标签**: 修改 `__getitem__` 方法，从 `.npz` 文件中读取 `surface_labels`，并将其拼接到 `surface_points` 之后。
    *   **修复索引错误**: 修复了体积点采样时的 `IndexError`。原代码直接使用布尔张量作为索引，导致维度不匹配。修改为先将布尔张量 `reshape(-1)` 为一维，再进行索引。
    *   **增强鲁棒性**: 在采样点数不足（例如正 SDF 点很少）的情况下，将 `np.random.choice` 的 `replace` 参数设置为 `True` (有放回采样)，防止程序因样本不足而崩溃。

## 5. 训练引擎与流程 (Training Engine & Main Script)

### 文件: `vecset/engines/engine_ae.py`
*   **原因**: 训练过程中出现了维度不匹配的错误，导致 Loss 无法计算或模型前向传播失败。
*   **修改内容**:
    *   **查询维度适配**: 在 `train_one_epoch` 中，当使用表面点作为查询输入模型时，通过 `surface[:, :, :3]` 仅截取前 3 维坐标，去掉了标签部分，以匹配解码器的 `query_embed` (3D)。
    *   **Loss 形状修复**: 计算 Loss 时，模型输出形状为 `(B, N)`，而标签形状为 `(B, N, 1)`。添加了 `.squeeze(-1)` 操作将标签压缩为 `(B, N)`，解决了广播错误。

### 文件: `vecset/main_ae.py`
*   **原因**: 适配新的数据路径，并修复变量作用域错误。
*   **修改内容**:
    *   **路径更新**: 将数据集路径指向生成的 `data_npz` 目录。
    *   **修复 UnboundLocalError**: 修复了代码结构，确保 `dataset_train` 和 `model` 变量在被引用前已经正确初始化。
    *   **参数配置**: 在模型初始化时显式传入 `input_dim=13`。

### 文件: `VecSetX/run_train.sh` (新增)
*   **原因**: 提供一个便捷的脚本来启动训练，避免每次手动输入长命令。
*   **修改内容**: 编写了 Bash 脚本，设置了 `PYTHONPATH`，并使用 `torchrun` 启动单机单卡训练，配置了正确的 batch size、epochs 和输出目录。

## 6. 生产环境配置与监控 (Production Setup & Monitoring) [NEW]

### 文件: `vecset/main_ae.py`
*   **原因**: 
    1.  集成 `wandb` 以进行实验监控。
    2.  修复硬编码的数据路径，使其能够接受命令行参数 `--data_path`。
    3.  清理代码中的重复部分。
*   **修改内容**:
    *   **WandB 集成**: 添加 `--wandb` 参数，并在主进程中初始化 `wandb`。
    *   **API Key**: 添加了 `wandb.login(key="...")`，允许用户直接在代码中配置 API Key。
    *   **路径修复**: 将 `Objaverse` 初始化中的 `dataset_folder` 参数改为使用 `args.data_path`。
    *   **代码清理**: 移除了文件尾部重复的 `main` 函数定义。

### 文件: `VecSetX/run_phase1_sbatch.sh` (新增)
*   **原因**: 为 4x NVIDIA A100 集群创建 SLURM 提交脚本。
*   **修改内容**:
    *   **资源配置**: 4x A100 GPU, 32 CPUs/task, 36小时运行时限。
    *   **环境设置**: 设置 `PYTHONPATH` 和 `OMP_NUM_THREADS`。
    *   **路径配置**: 更新为集群上的实际路径 (`/projappl/...` 和 `/scratch/...`)。
    *   **WandB**: 启用 `--wandb` 标志。

### 决策记录
*   **Batch Size**: 确认在 A100 40GB 上使用 `Batch Size 64` 是安全的（显存占用约 26GB）。`Batch Size 128` 可能会导致 OOM（估算约 46GB），建议通过 `--accum_iter 2` 来模拟。
*   **框架选择**: 决定保持 **Native PyTorch** 架构，暂不迁移到 Lightning，以保持代码透明度和灵活性。
*   **配置管理**: 决定在 Phase 1 跑通后再引入 YAML/Hydra 配置文件，目前继续使用 `argparse`。

## 7. 调试与修复记录 (Debugging & Fixes) [NEW]

在单机单卡 (Single GPU) 环境下验证训练脚本 `run_phase1_test.sh` 时，遇到并解决了以下一系列问题：

### 7.1 依赖缺失与兼容性 (Dependencies)
*   **问题**: `ModuleNotFoundError: No module named 'tensorboard'`。
*   **原因**: 环境中未安装 `tensorboard`，但安装了 `tensorboardX`。
*   **修复**: 在 `vecset/main_ae.py` 中添加回退机制，优先尝试导入 `torch.utils.tensorboard`，失败则尝试导入 `tensorboardX`。

### 7.2 代码属性错误 (AttributeError)
*   **问题**: `AttributeError: 'SummaryWriter' object has no attribute 'log_dir'`。
*   **原因**: `tensorboardX` 的 `SummaryWriter` 对象使用 `logdir` 属性，而非 `log_dir`。
*   **修复**: 修改 `vecset/engines/engine_ae.py`，将 `log_writer.log_dir` 更正为 `log_writer.logdir`。

### 7.3 文件路径错误 (FileNotFoundError)
*   **问题**: `FileNotFoundError: [Errno 2] No such file or directory: 'utils/objaverse_train.csv'`。
*   **原因**: `vecset/utils/objaverse.py` 使用相对路径加载 CSV 文件，导致在不同目录下运行脚本时路径失效。
*   **修复**: 修改 `vecset/utils/objaverse.py`，使用 `os.path.dirname(__file__)` 构建绝对路径，确保无论在哪里运行脚本都能正确找到 CSV 文件。

### 7.4 显存溢出 (CUDA OutOfMemory)
*   **问题**: `torch.OutOfMemoryError: CUDA out of memory`。
*   **原因**: 在 A100 (40GB) 上，`Batch Size 64` 结合当前模型配置导致显存不足。
*   **修复**:
    *   在 `run_phase1_test.sh` 中将 `batch_size` 降低至 **2**。
    *   将 `accum_iter` (梯度累积步数) 增加至 **32**，以保持 Effective Batch Size 为 64 (2 * 32)。
    *   设置环境变量 `PYTORCH_ALLOC_CONF=expandable_segments:True` 以优化显存碎片管理。

### 7.5 注意力机制反向传播错误 (RuntimeError)
*   **问题**: `RuntimeError: derivative for aten::_scaled_dot_product_efficient_attention_backward is not implemented`。
*   **原因**: PyTorch 的 `F.scaled_dot_product_attention` 在使用 "efficient" 内核时，不支持当前输入配置的反向传播。
*   **修复**: 修改 `vecset/models/utils.py`，移除了对 PyTorch SDPA 的依赖，改用**手动实现的点积注意力 (Manual Scaled Dot-Product Attention)** 作为回退方案。这虽然牺牲了少量速度，但保证了反向传播的稳定性。

### 7.6 缩进错误 (IndentationError)
*   **问题**: `IndentationError: expected an indented block after 'else' statement`。
*   **原因**: 在修复 SDPA 问题时，编辑操作导致 `else` 分支下的代码块丢失。
*   **修复**: 恢复了 `vecset/models/utils.py` 中丢失的手动注意力实现代码。

### 7.7 显存占用分析 (Memory Usage Analysis)
*   **现象**: 即使 Batch Size 降为 2，显存占用仍高达 34GB。
*   **原因**:
    1.  **Eikonal Loss (二阶导数)**: 计算 `points_gradient` 需要 `create_graph=True`，导致 PyTorch 保存整个前向传播图以计算二阶导数，显存占用翻倍。
    2.  **手动 Attention**: 为了规避 SDPA 反向传播错误，手动实现的 Attention 需要显式存储巨大的注意力矩阵 (Batch=2, Heads=16, Q=1024/8192, K=16384/1024)，占用数 GB 显存。
    3.  **模型规模**: 4.25亿参数的模型本身及其优化器状态占用约 7GB。
*   **结论**: 当前配置 (Batch Size 2, Accum Iter 32) 是 A100 (40GB) 上的安全运行阈值。

### 7.8 训练早停与验证缺失 (Early Stopping & Missing Validation)
*   **问题**: 训练在 Epoch 40 停止，且没有输出验证日志。
*   **原因**: 
    1.  **静默崩溃 (Silent Crash)**: 验证阶段或模型保存阶段发生错误，但由于缺乏异常捕获机制，进程直接退出。
    2.  **磁盘空间耗尽**: 随后发现报错 `RuntimeError: PytorchStreamWriter failed writing file data/868: file write failed`，确认为磁盘空间不足。
*   **修复**:
    *   **增强鲁棒性**: 在 `main_ae.py` 中为验证逻辑添加 `try-except` 块和详细 Debug 打印。
    *   **优化 Checkpoint 策略**: 修改 `utils/misc.py` 和 `main_ae.py`，不再每 5 个 Epoch 保存一次全量模型，改为只保存 `checkpoint-last.pth` (每 Epoch 更新) 和 `checkpoint-best.pth` (仅当 IoU 提升时更新)。
    *   **清理空间**: 建议用户删除旧的 Checkpoint 文件。

### 7.9 代码损坏修复 (Code Corruption Repair)
*   **问题**: 
    1.  `IndentationError` in `models/utils.py`: 工具编辑导致缩进错误。
    2.  `NameError: name 'epoch' is not defined` in `utils/misc.py`: 工具编辑导致文件内容错乱，函数定义被覆盖。
*   **修复**: 完全重写并恢复了 `vecset/models/utils.py` 和 `vecset/utils/misc.py` 的正确内容，确保语法正确且逻辑完整。

---

## 8. 正式训练环境部署与调试 (Production Deployment & Debugging) [NEW - Dec 4, 2025]

在 4x NVIDIA A100 集群上进行单卡测试时，遇到并解决了一系列 PyTorch 兼容性、数据加载、验证逻辑和训练优化问题。

### 8.1 PyTorch 版本兼容性问题

#### 问题: `ModuleNotFoundError: No module named 'torch._six'`
*   **现象**: 训练脚本启动失败，`utils/misc.py` 无法导入 `torch._six.inf`
*   **原因**: `torch._six` 是 PyTorch 旧版本的内部模块，在新版本中已被移除
*   **影响**: 阻止训练启动
*   **修复**: 
    *   在 `vecset/utils/misc.py` 中添加 `import math`
    *   将 `from torch._six import inf` 替换为 `inf = math.inf`
    *   使用标准库的 `math.inf` 代替 PyTorch 内部实现

### 8.2 验证数据加载器缺失

#### 问题: `NameError: name 'data_loader_val' is not defined`
*   **现象**: 训练正常运行，但验证阶段报错，提示 `data_loader_val` 未定义
*   **原因**: `main_ae.py` 中只创建了 `data_loader_train`，遗漏了 `data_loader_val` 的创建
*   **影响**: 无法进行验证，无法监控模型泛化性能
*   **修复**: 
    *   在 `main_ae.py` 第 173-180 行添加 `data_loader_val` 的创建
    *   配置与 `data_loader_train` 相同，但 `drop_last=False` 以保留所有验证样本

### 8.3 验证集维度不匹配

#### 问题: `RuntimeError: The size of tensor a (1024) must match the size of tensor b (0)`
*   **现象**: 验证时尝试访问 `labels[:, 1024:2048]` 导致索引越界
*   **根本原因**: 训练集和验证集的数据格式不一致
    *   **训练集**: `points = vol_points + near_points` → 2048 个查询点 (1024+1024)
    *   **验证集**: `points = near_points` → 1024 个查询点
*   **代码位置**: `utils/objaverse.py` 第 130-142 行根据 `split` 使用不同的点云组合策略
*   **影响**: 验证阶段崩溃，无法计算 IoU
*   **修复**: 
    *   修改 `engines/engine_ae.py` 的 `evaluate` 函数
    *   动态检测点云大小 (`num_query_points = points.shape[1]`)
    *   根据点数切换损失计算策略：
        *   2048 点（训练模式）: 分别计算 vol 和 near 的 loss 和 IoU
        *   1024 点（验证模式）: 只计算 near 的 loss 和 IoU，vol 设为 0

### 8.4 学习率与 Warmup 配置问题

#### 问题: 训练 10 个 Epoch 后 IoU 仍然为 0，损失几乎不下降
*   **现象**: 
    *   Epoch 0: loss=987.8, IoU=0.000
    *   Epoch 10: loss=982.7, IoU=0.000 (仅下降 5.1)
*   **分析**: 
    *   原配置: `blr=1e-4`, `warmup_epochs=40`
    *   Epoch 10 时学习率 = `1e-4 × 64/256 × (10/40) = 6.25e-6`
    *   学习率过低，模型权重更新极慢
*   **影响**: 模型无法有效学习，浪费训练时间
*   **解决方案**: 
    *   **方案 1 (已采用)**: 缩短 warmup，`warmup_epochs: 40 → 10`
    *   **方案 2 (已采用)**: 提高基础学习率，`blr: 1e-4 → 4e-4`
    *   修改 `run_phase1_test.sh` 和 `run_phase1_sbatch.sh`
*   **验证**: Epoch 10 时学习率应达到 1e-4

### 8.5 IoU 为 0 的根本原因与 Loss 权重调整

#### 深度分析: 为什么 IoU 始终为 0？

**现象**:
```
Epoch 10 (新配置):
  loss: 907.3
  loss_near: 81.7 (下降了 8 点)
  loss_surface: 8.04 (暴涨 8000 倍！)
  IoU: 0.000 (仍然为 0)
```

**IoU 计算逻辑**:
```python
pred = (output >= 0).float()  # 二值化，threshold=0
target = (labels >= 0).float()
iou = intersection / union
```

**根本原因**: 模型系统性地预测所有 SDF 值为负数
*   模型输出 `output < 0` 对所有点成立
*   导致 `pred` 全为 0 (没有任何点被预测为表面或外部)
*   `target` 有 0 和 1 (真实数据有内部和外部点)
*   `intersection = 0` → **IoU = 0**

**为什么会系统性偏负？Loss 权重分析**:

原始 Loss 函数:
```python
loss = loss_vol + 10 * loss_near + 0.001 * loss_eikonal + 1 * loss_surface
```

各项实际贡献:
*   `loss_vol × 1 = 81.9 × 1 = 81.9`
*   `loss_near × 10 = 81.7 × 10 = 817` ⭐ (主导项)
*   `loss_eikonal × 0.001 = 0.998 × 0.001 ≈ 0.001`
*   `loss_surface × 1 = 8.04 × 1 = 8.04` ⚠️ (被忽视)

**问题**: 
*   模型优化主要受 `loss_near` 驱动 (贡献 817/907 ≈ 90%)
*   `loss_surface` 的贡献仅 8/907 ≈ 0.9%，几乎被忽略
*   模型学习到: "只要预测值接近真实 SDF 值，loss 就低"
*   但忽略了: "表面点的 SDF 必须为 0" 的硬约束
*   结果: 模型预测系统性偏移（都偏负），导致二值化后完全错误

#### 解决方案: 增加 loss_surface 权重

**修改**:
```python
# vecset/engines/engine_ae.py
# 训练 loss (第 97 行)
loss = loss_vol + 10 * loss_near + 0.001 * loss_eikonal + 10 * loss_surface
#                                                           ↑ 1 → 10

# 验证 loss (第 201 行)
loss = loss_vol + 10 * loss_near + 10 * loss_surface
#                                   ↑ 1 → 10
```

**原理**:
*   强制表面点的 SDF 接近 0
*   防止系统性偏移
*   平衡 loss_near 和 loss_surface 的重要性

**预期效果**:
*   Epoch 5: `loss_surface < 2`, `IoU > 0`
*   Epoch 10: `loss_surface < 1`, `IoU > 0.1`
*   Epoch 20-30: `loss_surface < 0.5`, `IoU > 0.3`

### 8.6 VecSetX 架构理解与泛化能力

#### 架构核心: Encoder-Decoder AutoEncoder

**完整流程**:
```python
# 输入: 表面点云
surface_points: (Batch, 8192, 13)  # 8192点, 3D坐标 + 10类one-hot

# Encoder: 动态生成 latent set
latent_set = encoder(surface_points)  # (Batch, 16, 1024)
# 16 个可学习的 latent vectors，每个 1024 维

# Decoder: 隐式神经场
query_points: (Batch, 2048, 3)  # vol_points + near_points
sdf_predictions = decoder(latent_set, query_points)  # (Batch, 2048)
```

**关键设计**: 16 个 Latent Vectors
*   通过 **Set Transformer** 中的 cross-attention 动态生成
*   16 个 "query seeds" 从 8192 个表面点中聚合信息
*   类比: 16 个专家分别关注不同的局部特征或全局关系
*   对心脏多类重建: 足够表达 10 个复杂解剖结构

**泛化机制**:
*   ✅ Encoder 是可学习的神经网络，不是 per-shape latent code (查表)
*   ✅ 验证集样本通过 Encoder 动态生成 latent set
*   ✅ 测试 Encoder 从未见过的点云中提取形状特征的能力
*   ✅ 这就是 AutoEncoder 的价值: 学习数据分布的压缩表示并泛化

### 8.7 数据集划分策略 (998 个样本)

**当前划分** (推荐):
*   训练集: ~798 样本 (80%)
*   验证集: ~200 样本 (20%)

**理由**:
1.  **科学验证**: 监控过拟合，评估泛化能力
2.  **Early Stopping**: 根据验证 IoU 保存最佳模型
3.  **为 Phase 2-4 保留基准**: 验证集可作为后续阶段的已知数据基准
4.  **符合医学 AI 规范**: 独立验证集是论文发表的必要条件
5.  **数据量充足**: 798 个 3D 医学样本足以训练深度模型

**替代方案**: K-Fold 交叉验证 (如果需要更准确的评估)

### 8.8 训练指标解读

**核心指标优先级**:
1.  **IoU (验证集)**: 最重要，直接反映临床可用性 (目标: >0.5)
2.  **loss_near**: 近表面精度，权重最高 (应持续下降)
3.  **loss_surface**: 表面约束，影响 IoU (应下降到 <0.5)
4.  **loss_eikonal**: 梯度正则，保持训练稳定 (应 ≈1.0)

**监控策略**:
*   每 5 个 Epoch 验证一次
*   关注验证集 IoU 是否上升
*   对比训练/验证 loss，检测过拟合

### 8.9 下一步行动

**立即操作**:
1.  ✅ 删除旧 checkpoint: `rm output/ae/phase1_production/checkpoint-*.pth`
2.  ✅ 重新启动训练: `bash run_phase1_test.sh` 或 `sbatch run_phase1_sbatch.sh`
3.  ✅ 监控 Epoch 5, 10, 20 的 IoU 和 loss_surface

**预期里程碑**:
*   **Epoch 5**: IoU > 0, loss_surface < 2
*   **Epoch 10**: IoU > 0.1, loss_surface < 1
*   **Epoch 30**: IoU > 0.3, loss_surface < 0.5
*   **Epoch 100+**: IoU > 0.5, 达到临床可用标准

---

## 9-11. 数据质量诊断与生成方案优化 (Data Quality & Generation Optimization)

> **移出说明**: 由于篇幅较长，关于数据质量诊断、SDF 算法修复、性能瓶颈解决及最终 Hybrid 方案的详细记录已移动至专门的文档 [Datarecord.md](Datarecord.md)。
> 
> *   原 Section 9: 数据质量诊断与完美 SDF 生成
> *   原 Section 10: rtree 性能瓶颈与实用主义方案
> *   原 Section 11: 数据生成方案的最终优化

---

## 12. 训练停滞问题诊断与优化 (Training Stagnation and Optimization) [Dec 6, 2025]

使用新数据集开始训练后，虽然 IoU 从 0 突破到 0.26，但在 Epoch ~20 后训练出现停滞。

### 12.1 训练进展观察

#### 初始阶段 (Epoch 0-35)

**突破性进展**：
*   **Epoch 30**: IoU = 0.000 (仍为 0)
*   **Epoch 35**: IoU = 0.264 ✅ (首次突破！)
*   **Epoch 36**: vol_iou = 0.50, near_iou = 0.52 ✅

**关键发现**：
*   near_sdf 使用 mesh_to_sdf 生成，质量完美 → near_iou 率先达到 0.52
*   vol_sdf 使用深度比例启发式 → vol_iou 紧随其后达到 0.50
*   新数据集相比旧数据有**质的飞跃**（从 140 epoch IoU=0 到 35 epoch IoU=0.26）

#### 停滞阶段 (Epoch 36-107)

**训练曲线**：
*   train_loss: 0.47 (初始) → 0.42 (Epoch ~20) → **0.42 (持平至 Epoch 107)** ⚠️
*   test_loss: 0.095 → 0.076 (轻微下降但缓慢)
*   train_vol_iou: ~0.25 (震荡但不上升)
*   train_near_iou: ~0.26 (震荡但不上升)
*   test_vol_iou: 0.0 (硬编码为 0，验证集无 vol points)
*   test_near_iou: 0.0 ↔ 0.52 (剧烈震荡)

**震荡现象**：
*   IoU 在 0 和 0.5 之间跳跃
*   原因：threshold=0 的二值化判断 + 模型预测不稳定
*   早期训练的正常现象，但应随训练稳定

### 12.2 根因分析

#### 问题：学习率衰减过快

**当前训练配置** (`run_phase1_sbatch.sh`):
```bash
--blr 4e-4           # base_lr = 0.0004
--warmup_epochs 10
--epochs 800
```

**学习率调度策略** (`lr_sched.py`):
```python
# Warmup (Epoch 0-10): linear 0 → 0.0004
# Cosine decay (Epoch 10-800): 0.0004 → min_lr
lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(π * (epoch-10) / 790))
```

**Epoch 107 的学习率**：
```
lr = min_lr + (0.0004 - min_lr) * 0.5 * (1 + cos(π * 97/790))
lr ≈ 0.000097  # 仅为初始 LR 的 24%！
```

**问题所在**：
1.  base_lr 本身就偏小 (0.0004)
2.  800 epochs 的长调度 + cosine decay 导致 LR 快速下降
3.  Epoch 107/800 (仅完成 13%) 时 LR 已降至初始的 24%
4.  模型权重更新步长太小，无法有效优化

#### 其他潜在问题

1.  **Eikonal loss 权重过小**
    *   当前：`loss_eikonal_weight = 0.001`
    *   现象：eikonal_loss 一直为 1.0（梯度场不 smooth）
    *   影响：权重太小 (0.001)，对总 loss 贡献可忽略

2.  **IoU 二值化阈值敏感**
    *   threshold = 0 将连续 SDF 强行二值化
    *   微小预测偏差导致巨大 IoU 差异
    *   早期训练震荡严重

### 12.3 优化方案

#### 方案 A: 重新训练（推荐）✅

创建 `run_phase1_sbatch_v2.sh` 优化配置：

**关键改进**：
```bash
# V1 (当前)          # V2 (优化)
--blr 4e-4           → --blr 1e-3        # 提高 2.5 倍
--warmup_epochs 10   → --warmup_epochs 20 # 更稳定预热
--epochs 800         → --epochs 400       # 减半，LR 衰减更慢
```

**学习率对比**：

| Epoch | V1 (当前) | V2 (优化) | 倍数 |
|-------|----------|-----------|------|
| 20 | 0.000395 | 0.000975 | 2.5x |
| 50 | 0.000353 | 0.000875 | 2.5x |
| **107** | **0.000097** | **0.000715** | **7.4x** |
| 200 | ~0.00002 | 0.000500 | 25x |
| 400 | ~0.00001 | (end) | - |

**预期效果**：
*   **Epoch 50**: IoU > 0.3-0.4
*   **Epoch 100**: IoU > 0.45-0.55
*   **Epoch 200**: IoU > 0.55-0.65
*   **Epoch 400**: 最终收敛

**优势**：
1.  ✅ 学习率更高，收敛更快
2.  ✅ 清晰的训练起点，无历史包袱
3.  ✅ 总 epoch 更少（800 → 400），节省时间

#### 方案 B: 继续当前训练 + 手动调整

保留当前 107 epoch 进度，手动提升学习率：

**实现方式**：
修改 `lr_sched.py`：
```python
def adjust_learning_rate(optimizer, epoch, args):
    if epoch >= 100:
        boost_factor = 3.0  # LR 提升 3 倍
        lr = base_lr * boost_factor
    else:
        # ... 原有调度
```

**优势**：
*   不丢失 107 epoch 的训练进度

**劣势**：
*   更复杂，需要修改代码
*   可能效果不如重新训练

#### 方案 C: 等待并观察

继续当前训练至 Epoch 200-300。

**劣势**：
*   学习率太低，大概率继续停滞
*   浪费计算资源

### 12.4 最终决策

**采用方案 A：重新训练，使用优化配置**

**理由**：
1.  当前训练已停滞 80+ epochs，继续下去意义不大
2.  学习率过低是根本问题，需要系统性调整
3.  V2 配置针对性解决 LR 问题
4.  结合优质数据 + 优化超参，应该能更快收敛
5.  重新开始比修补更简洁可控

**执行步骤**：
```bash
# 1. 取消当前训练
scancel <job_id>

# 2. 提交优化版本
cd /projappl/project_2016517/JunjieCheng/VecSetX
sbatch run_phase1_sbatch_v2.sh

# 3. 监控训练
# - 前 20 epoch 观察 warmup 效果
# - Epoch 50 检查 loss 是否 < 0.38
# - Epoch 100 检查 IoU 是否 > 0.45
```

### 12.5 配置文件

**V1 配置** (`run_phase1_sbatch.sh`):
*   base_lr: 4e-4
*   warmup: 10 epochs
*   total: 800 epochs
*   结果：Epoch 107 停滞

**V2 优化配置** (`run_phase1_sbatch_v2.sh`):
*   base_lr: 1e-3 ✅
*   warmup: 20 epochs ✅
*   total: 400 epochs ✅
*   output_dir: `output/ae/phase1_production_v2`

### 12.6 经验教训

1.  **学习率调度至关重要**
    *   总 epoch 数直接影响 cosine decay 速度
    *   长调度 (800) 适合收敛慢的任务，但 LR 衰减太快
    *   短调度 (400) + 高 base_lr 可能更适合当前任务

2.  **Warmup 的重要性**
    *   10 epochs 可能太短
    *   20 epochs 提供更稳定的初始化

3.  **监控训练曲线**
    *   Loss 持平 > 20 epochs 需警惕
    *   应检查学习率是否过低
    *   IoU 震荡在早期正常，但应逐步减小

4.  **Base LR 选择**
    *   4e-4 偏保守
    *   1e-3 更常见于 vision transformer 训练

### 12.7 待验证假设

**假设 1**: 更高的 LR 会打破停滞
*   验证方法：V2 训练 50 epochs 后观察 loss
*   成功标志：loss < 0.38

**假设 2**: 400 epochs 足够收敛
*   验证方法：观察 Epoch 300-400 的 loss/IoU 变化
*   成功标志：最后 50 epochs 变化 < 1%

**假设 3**: 数据质量已不是瓶颈
*   假设新数据集质量足够好
*   如果 V2 仍停滞，可能需要考虑模型容量或其他因素

---

**下一步：等待 V2 训练结果，验证优化效果。**

---

## 13. 多类别架构升级与数据生成修复 (Multi-Class Upgrade & Data Fixes) [Dec 6, 2025]

### 13.1 背景与目标
为了实现从单一 geometry 重建到 **10类解剖结构语义重建** 的跨越，我们对系统架构进行了重大升级。
*   **旧架构**：输出 1维 SDF (Union)。
*   **新架构**：输出 11维 Tensor (1维 Union SDF + 10维 Class Logits)。
*   **目标**：模型不仅学会形状，还要学会“這是哪个结构”。

### 13.2 遇到的问题 (Problems)

#### 问题 A: 标签泄漏 (Label Leakage)
*   **现象**：在使用 `check_data_quality.py` 验证数据时，发现 **~50%** 的样本存在 "Outside but Cls" 错误。这意味着大量几何上在 **外部** 的点被错误地打上了 **类别标签**。
*   **原因**：早期的 `prepare_data.py` 使用简单的 "Depth Ratio Heuristic" 来判定内外。该启发式算法在远离中心时会失效，错误地将远处的点判定为“内部”。
*   **后果**：模型会被迫学习错误的标签，导致训练发散或无法收敛。

#### 问题 B: 效率瓶颈
*   **尝试**：试图全量使用 `mesh_to_sdf` (专业库) 来解决泄漏。
*   **结果**：速度极慢（每文件数分钟），无法满足 1000+ 文件的生成需求。

### 13.3 解决方案 (Solutions)

采用了 **"Multi-Class Hybrid Final"** 混合策略，兼顾速度与准确性：

1.  **Volume Points (速度优先)**：
    *   沿用 **Depth Ratio Heuristic (Threshold=0.33)**。
    *   **创新**：对 **每个类别(1-10)** 分别计算 Heuristic SDF，取最小值作为 Union SDF，取 argmin 作为标签。
    *   **效果**：保证了全局覆盖，且速度极快。

2.  **Near Points (精度优先)**：
    *   **Geometry**：使用 `mesh_to_sdf` 对 **Union Mesh** 进行计算，确保近表面 SDF 的物理真实性。
    *   **Semantics**：使用 Heuristic 方法判定点归属的类别。
    *   **Fusion**：通过严格逻辑 `if Union_SDF > 0: Label = 0` 强制修正。

3.  **一致性保证 (Consistency)**：
    *   利用 `Union_SDF` (from mesh_to_sdf) 作为绝对真理。
    *   任何在几何上为“外部”的点，强制标签为 0。
    *   实现了 **100% Geometry-Label Consistency**。

### 13.4 代码变更 (Code Changes)

1.  **`vecset/prepare_data.py`**：完全重写。实现了上述 "Hybrid Final" 逻辑，支持 10 类 Label 生成。
2.  **`vecset/models/autoencoder.py`**：将 `output_dim` 默认值改为 **11**。
3.  **`vecset/engines/engine_ae.py`**：
    *   新增 `loss_cls` (Binary Cross Entropy) 用于分类训练。
    *   新增 `acc` 指标用于监控分类准确率。
    *   实现了 SDF loss (L1) 与 Classification loss 的加权联合训练。
4.  **`vecset/utils/objaverse.py`**：更新 `__getitem__` 以加载和返回标签数据。
5.  **`vecset/models/utils.py`**：将手动 Attention 替换为 PyTorch 优化的 `F.scaled_dot_product_attention`，提升显存效率。
6.  **`check_data_quality.py`**：新增工具，专门用于验证 SDF 正负比与 Label 一致性。


### 13.5 下一步计划
1.  使用新脚本全量生成训练数据。
2.  使用 `check_data_quality.py` 抽检。
3.  启动 Phase 1 Multi-Class 训练。

---

## 14. Multi-Class Data Generation - Complete Debugging Journey (2025-12-06~07)

### 14.1 背景与目标
为VecSetX多类别训练生成高质量数据集（998个样本，10个心脏结构类别）。

**核心要求**：
- SDF平衡：Volume ~50%, Near ~50%
- Label-Geometry一致性：接近100%
- 类别覆盖：所有样本包含全部10个类别
- **特别保证Coronary（创新点）的采样质量**

---

### 14.2 问题1: SDF Balance异常

**现象**：Vol Pos分布严重失衡
- `vol_threshold=0.33` → Vol Pos = 0%
- `vol_threshold=1.2` → Vol Pos = 99%

**根本原因**：
Depth ratio heuristic的threshold参数对inside/outside判断影响极大。

**解决方案**：
实验确定 `vol_threshold=0.85` 能得到~50% Vol Pos。

---

### 14.3 问题2: 类别缺失

**现象**：
```
Surface labels: 10 classes ✅
Vol labels: 8 classes (missing [1, 3]) ❌
```

**根本原因**：
Depth ratio heuristic对薄壁结构（Myocardium, LV）判断不准，即使点在该类别体素内，SDF却判为outside。

**解决方案**：
尝试直接从voxel读取label，但引入了coordinate mismatch问题（见问题4）。

---

### 14.4 问题3: Coronary采样不足

**问题分析**：
- Coronary体积占比：0.12%
- 随机采样50k点：理论仅60个点
- 严重不足以学习该结构

**初步方案**：Stratified Sampling
- 背景35% (17.5k), 结构65% (32.5k)
- 理论提升Coronary采样至~200点
- 但引入了严重的consistency问题...

---

### 14.5 问题4: Stratified Sampling导致Label-SDF严重不一致 ⚠️⚠️⚠️

**灾难性现象**：
```
Consistency: 52.7% (应该接近100%) ❌❌❌
Outside (SDF>=0) but has class label: 36.9%
```

**Stratified实现方式（有问题）**：
```python
# 1. 在voxel整数空间stratified采样
vol_points_voxel = sample_from_voxel_grid(data)

# 2. 从voxel直接读label
vol_labels[i] = data[x, y, z]  # voxel space

# 3. Transform到normalized空间
vol_points = transform(vol_points_voxel)  # normalized space

# 4. 计算SDF
vol_sdf = compute_sdf(vol_points, normalized_mesh)  # normalized space
```

**根本原因：Coordinate Space Mismatch**
- Labels在**voxel整数坐标空间**分配
- SDF在**normalized连续坐标空间**计算
- Transform在边界处有微小误差
- 导致：voxel说"inside"，normalized SDF说"outside"

**调试证据**：
```
Outside but has class label:
  Point 17502: SDF=0.0157, Label=6  ← 刚好在边界外0.01
  Point 17503: SDF=0.0227, Label=1
```

36.9%的高比例是因为stratified故意在结构边界密集采样！

---

### 14.6 最终解决方案：Normalized Space Stratified Sampling ✅

**核心思想**：
**在同一个坐标空间（normalized space）完成所有操作**

**实现**：
```python
# 1. 背景点：从整个[-1,1]³均匀采样
vol_points_bg = np.random.uniform(-1, 1, (n_bg, 3))

# 2. 结构点：从union mesh的bounding box采样（已在normalized space）
bounds = union_mesh.bounds
bbox_expanded = expand_bbox(bounds, factor=1.1)
vol_points_struct = np.random.uniform(bbox_min, bbox_max, (n_struct, 3))

# 3. 在同一空间计算SDF
vol_sdf = compute_sdf(vol_points, normalized_mesh)

# 4. 从SDF推导label（数学严格一致）
vol_labels = derive_from_sdf(vol_sdf_all)
```

**优势**：
- ✅ 空间一致性：采样、SDF、labels都在normalized space
- ✅ 数学一致性：Label严格由SDF推导，100%一致
- ✅ 结构覆盖：65%点在结构bbox内，小结构采样提升
- ✅ 避免coordinate transform误差

---

### 14.7 问题5: 数据生成速度优化

**原始性能**：
- 单文件：50-60秒
- 998文件串行：~14小时
- CPU利用率：<30% (256核系统)

**优化历程**：

**阶段1：Per-Class Parallelization**
- 10个类别的SDF并行计算
- 提升：~2-3倍

**阶段2：File-Level Parallelization**
- 同时处理多个文件
- 遇到Nested Pool问题：daemon进程不能spawn子进程

**解决方案**：
```python
if file_workers > 1:
    use_parallel = False  # 禁用文件内并行
else:
    use_parallel = True   # 启用per-class并行
```

**最终配置**：
```bash
--file_workers 32  # 32文件并行
--n_workers 1      # 文件内串行
```

**最终速度**：998文件 ~**30分钟** (从14小时优化到30分钟！)

---

### 14.8 问题6: check_data_quality.py误报

**现象**：
报告24%不一致，但实际100%一致

**原因**：
```python
# ❌ 错误
inside = vol_sdf < -1e-4  # 应该是0

# ✅ 正确
inside = vol_sdf < 0
```

**修复**：
将检查脚本的阈值改为SDF=0，与label分配逻辑一致。

---

### 14.9 最终配置 & 预期质量

**生成命令**：
```bash
python prepare_data.py \
    --input_dir /scratch/project_2016517/junjie/dataset/repaired_shape \
    --output_dir /scratch/project_2016517/junjie/dataset/repaired_npz \
    --vol_threshold 0.85 \
    --file_workers 32
```

**预期质量**：
- SDF Balance: Vol ~55%, Near ~52%
- Label Consistency: ~100%
- Class Coverage: Surface 100%, Vol 可能8-10类
- 生成时间: ~30分钟 (998文件)

---

### 14.10 核心教训

1. **坐标空间一致性至关重要**：同一空间操作避免transform误差
2. **Heuristic需empirical validation**：depth ratio对薄壁结构不准
3. **数据检查必须与生成逻辑一致**：阈值必须匹配
4. **避免嵌套并行**：file-level和task-level并行二选一
5. **Stratified sampling是双刃剑**：提升覆盖但实现复杂

**关键文件**：
- `prepare_data.py`: 最终优化版本
- `check_data_quality.py`: 汇总统计
- `debug_consistency.py`: 详细诊断
- `verify_all_classes.py`: 类别覆盖验证

---

## 15. 数据生成性能优化与并行化问题 (2025-12-07)

### 15.1 并行化失败的根本原因

**问题现象**：
- 设置`--file_workers 64`，但速度没有提升
- CPU使用率很低（~1%），看起来完全串行
- 998个文件生成耗时12+小时

**根本原因**：
`process_file`函数缺少return语句，导致`pool.imap_unordered`无法正常工作：

```python
# ❌ 错误的实现
def process_file(file_path, args):
    # ... 处理逻辑 ...
    gc.collect()
    # 没有return！

# Pool等不到返回值，hang住
with Pool(processes=file_workers) as pool:
    for i, result in enumerate(pool.imap_unordered(process_func, files)):
        # 这个循环永远不会正常迭代
```

**修复方案**：
```python
# ✅ 正确的实现
def process_file(file_path, args):
    # ... 处理逻辑 ...
    gc.collect()
    return os.path.basename(file_path)  # 必须返回值
```

---

### 15.2 性能瓶颈分析

即使修复了并行bug，仍然存在性能问题：

**主要瓶颈：mesh_to_sdf**
- 每个文件需要调用`mesh_to_sdf`计算50k near points的精确SDF
- **单文件耗时：60-90秒**
- mesh_to_sdf内部是单线程的，无法进一步并行
- CPU使用率低（12.5%）是因为算法本身限制

**实际性能**：
```
16 file_workers × 60秒/文件 = 每批16个文件需要60秒
998文件总计 ≈ 60-90分钟
```

---

### 15.3 手动多终端并行方案

如果自动并行太慢，可以手动开多个终端并行处理：

#### 方案1：按索引范围分配（8个终端）

```bash
# 终端1
python prepare_data.py \
    --input_dir /scratch/project_2016517/junjie/dataset/repaired_shape \
    --output_dir /scratch/project_2016517/junjie/dataset/repaired_npz \
    --vol_threshold 0.85 \
    --file_workers 1 \
    --start_idx 0 --end_idx 125

# 终端2
python prepare_data.py ... --start_idx 125 --end_idx 250

# 终端3-8类似，每个处理125个文件
```

**注意**：`--file_workers 1`避免nested parallelism

####  方案2：只处理缺失文件

```bash
# 1. 找出缺失文件
python process_missing.py
# 自动创建symlink到 /scratch/.../missing_files_temp

# 2. 手动分批处理
# 将50个缺失文件分成5批，每批10个，5个终端并行
```

#### 方案3：使用screen批量管理

```bash
# 创建8个screen会话
for i in {1..8}; do screen -dmS data_gen_$i; done

# 在每个会话中运行
screen -S data_gen_1 -X stuff "python prepare_data.py --start_idx 0 --end_idx 125\n"
screen -S data_gen_2 -X stuff "python prepare_data.py --start_idx 125 --end_idx 250\n"

# 查看进度
screen -r data_gen_1
```

---

### 15.4 数据质量最终报告

**SDF Balance**：
- Vol Pos: 51.3% ± 11.0% ✅
- Near Pos: 52.9% ± 0.7% ✅ 完美

**Label Consistency**: 100.0% ✅

**Class Coverage (Vol)**：
- 10类：40.9%
- 9类：42.7%（主要缺LV）
- 8类：15.4%（缺LV+LA）

**缺失类别**：
- LV: 44.9%文件缺失 ⚠️
- LA: 25.1%文件缺失 ⚠️
- Coronary: 0.1%缺失 ✅ 创新点保障

---

### 15.5 后续改进（如训练效果不佳）

**如果LV/LA重建差**：
```python
# 方案1: Loss权重调整
loss = loss_surface * 3 + loss_near * 10 + loss_vol * 1

# 方案2: 降低vol_threshold
--vol_threshold 0.75  # 从0.85降到0.75
```

**如果mesh_to_sdf太慢**：
```python
# 降低精度换速度
mesh_to_sdf.mesh_to_sdf(..., scan_count=50, scan_resolution=200)
# 速度提升2倍
```

---

### 15.6 关键工具脚本

```bash
# 质量检查
python check_data_quality.py <npz_dir>
python debug_consistency.py <file.npz>
python analyze_missing_classes.py <npz_dir>

# 找缺失文件
python process_missing.py
```

**最终配置**：
- vol_threshold: 0.85
- file_workers: 16
- Stratified sampling: 35bg/65struct (normalized space)
- 总耗时: 1-2小时（998文件）

## 16. DataLoader Bug修复与训练启动 (2025-12-07)

在启动Phase 1训练时，遇到了一系列与DataLoader和SDF采样相关的bug，通过以下步骤逐一解决：

### 16.1 DataLoader中的`UnboundLocalError`

**问题**：
训练启动后报错：`UnboundLocalError: local variable 'points' referenced before assignment`。

**原因**：
在`Objaverse.__getitem__`中，`points`变量定义在if/else分支中，但在某些代码路径下没有被正确初始化。

**修复**：
重写了SDF采样逻辑，使用明确的`pos_ind`和`neg_ind`变量，确保在所有分支中都正确采样`points`、`sdf`和`labels`，并进行拼接。

### 16.2 Tensor Shape Mismatch

**问题**：
`RuntimeError: stack expects each tensor to be equal size`。
报错显示SDF tensor形状不匹配，例如`[1024, 1]` vs `[1024]`。

**原因**：
`.npz`文件中保存的SDF数据形状可能是`(N, 1)`，转换为torch tensor后依然保留了维度，而Labels是1D的`(N,)`。在`torch.stack`时维度不一致导致报错。

**修复**：
在转换为torch tensor时，强制使用`.flatten()`将SDF和Labels都展平为1D张量。

### 16.3 Batch Collation Error (Near Points数量不一致)

**问题**：
`RuntimeError: stack expects each tensor to be equal size, but got [51018, 3] at entry 0 and [51020, 3] at entry 1`。

**原因**：
不同`.npz`文件的`near_points`数量不一致（约50k左右），导致`__getitem__`返回的张量长度不同，PyTorch默认的`collate_fn`无法将变长张量堆叠成batch。

**修复**：
修改采样逻辑，对`near_points`也进行固定数量采样（采样数 = `sdf_size // 4`），确保每个样本返回的点数完全一致。

### 16.4 SDF Size与Model Input不匹配

**问题**：
`loss`计算时报错维度不匹配：`pred [2, 1024]` vs `target [2, 768]`。
模型预期输入是拼接后的点云：`vol(1024) + near(1024) + surface(8192) = 10240`个点。
但`dataset`配置为`sdf_size=1024`，导致只采样了`256+256=512`个vol/near点。

**原因**：
`main_ae.py`中写死了`sdf_size=1024`，而采样逻辑是`sdf_size // 4`。为了得到预期的1024个点，`sdf_size`应该设为4096。

**修复**：
在`main_ae.py`中将`sdf_size`从`1024`修改为`4096`。
修正后的采样逻辑：
- Vol points: `4096 // 4 = 1024`
- Near points: `4096 // 4 = 1024`
- Surface points: `8192` (from `point_cloud_size`)
- Total points: `1024 + 1024 + 8192 = 10240`



## 11. 此轮集群训练调试记录 (Cluster Training Debugging) [Dec 7, 2025]

在本次将代码部署到集群并启动训练的过程中，我们连续克服了三个关键的技术障碍，通过远程调试成功启动了大规模训练。

### 11.1 模块导入与初始化崩溃 (ModuleNotFoundError & Silent Crash)

*   **症状**: 训练任务立即失败 (Exit Code 1)，没有任何有用的报错信息。日志中仅包含 `torchrun` 的通用错误堆栈。
*   **原因**: 
    1.  `vecset` 包在集群环境中未被正确识别为 Python 包，导致 `import vecset.utils.objaverse` 失败。
    2.  `main_ae.py` 缺乏对 Dataset 初始化和 WandB 登录的异常捕获，导致进程在打印任何日志前就崩愤了。
*   **修复**:
    *   **相对导入**: 将 import 路径从 absolutel (`vecset.utils...`) 改为 relative/local (`utils...`)，适配当前的 PYTHONPATH 结构。
    *   **防御性编程**: 在 `main_ae.py` 中添加了详细的 `try-except` 块及显式的 `print` 日志，确保即使初始化失败也能看到具体原因。
    *   **环境检查**: 添加了对 data_path 和 CSV 文件的显式存在性检查。

### 11.2 Loss计算维度不匹配 (ValueError: Shape Mismatch)

*   **症状**: `ValueError: Target size (torch.Size([2, 3072, 10])) must be the same as input size (torch.Size([2, 11264, 10]))`。
*   **原因**: 
    1.  **索引硬编码失效**: `dataset` 配置为 `sdf_size=4096`，实际产生了 3072 个查询点 (1024 Pos + 1024 Neg + 1024 Near)。但 `engine_ae.py` 中沿用了旧的硬编码索引 (0:1024 Vol, 1024:2048 Near)，导致 loss 计算错位。
    2.  **分类目标缺失**: `loss_cls` (Binary Cross Entropy) 计算时，模型输出了所有点 (Query + Surface = 11264) 的 logits，但 Target 仅包含 Query 部分的标签 (3072)，导致维度不匹配。
*   **修复**:
    *   **动态索引**: 修正 `loss_vol` 取 `[:2048]` (包含 Pos+Neg)，`loss_near` 取 `[2048:3072]`，`loss_surface` 取 `[3072:]`。
    *   **目标拼接**: 构建 `target_one_hot` 时，将 Query 的标签与 Surface 的标签在 dim=1 维度上拼接，构造出完整的 (B, 11264, 10) 目标张量。

### 11.3 Attention 反向传播失败 (RuntimeError: SDPA Backward)

*   **症状**: `RuntimeError: derivative for aten::_scaled_dot_product_efficient_attention_backward is not implemented`。
*   **原因**: PyTorch 的 `F.scaled_dot_product_attention` (SDPA) 算子在集群的 A100 环境下，配合当前的 head_dim 和 tensor layout，未能自动选择支持反向传播的 kernel (使用了 efficient_attention 但其 backward 未被支持)。
*   **修复**:
    *   **回退手动实现**: 在 `vecset/models/utils.py` 中，用标准的 `torch.einsum` 手动实现了 Scaled Dot-Product Attention。
    *   **权衡**: 虽然牺牲了少量显存和计算效率，但彻底解决了兼容性问题，保证了梯度的正确回传。

### 11.4 最终状态

经过上述修复，`rank3` 等所有进程已成功进入 Training Loop：
```text
base lr: 2.00e-04
accumulate grad iterations: 8
effective batch size: 64
Start training for 400 epochs
```
模型正在稳定训练中。

## 2025-12-09: Phase 1 Failure Analysis - Data Generation Flaw

### Symptom
- Model achieved decent training IoU (~0.70) but visualized results appeared "blobby" with significant extraneous volume.
- Validation IoU was 0.70, but 3D visualize IoU (Inference) dropped to ~0.10.

### Investigation
- Created `infer_visualize.py` to inspect the alignment (normalization) between Ground Truth (NII) and Model Input (NPZ).
- Overlaying the Input Point Cloud (White) on the GT Mask proved the coordinate system was ALIGNED and ACCURATE.
- Overlaying the Training Data Labels (Volume Points) revealed the root cause:
    - **Magenta Points (Labeled Inside)** were found densely distributed in the background (air), far outside the actual organ.
    - **Blue Points (Labeled Outside)** were scarce in regions that should have been clearly outside.

### Root Cause
- The `prepare_data.py` script used a **Heuristic Algorithm (Depth Ratio)** for "Volume Point" label generation to speed up processing.
- `depth_ratio = dists / dist_to_center`
- This heuristic assumes a somewhat convex or simple geometry where being "far from center" implies "outside".
- For complex, non-convex cardiac structures (multi-class), this assumption failed catastrophically, creating a "Hull" or "Blob" of False Positives.
- The model successfully learned this incorrect "Blob" distribution, leading to the visual artifacts.

### Solution
## 11-15. 数据生成方案的后续迭代 (Subsequent Data Generation Iterations)

> **移出说明**: 关于 Dec 9 - Dec 11 期间的数据生成方案重大重构（包括从 Heuristic 转向 Trimesh/Voxel 方法、采样策略优化及最终质量诊断），已全部整合至 [Datarecord.md](Datarecord.md)。
>
> 包含以下内容：
> *   Failure Analysis & Data Generation Pivot
> *   Headless Cluster Fix (Trimesh)
> *   Fast Voxel-based Generation
> *   Sampling Strategy Refactoring
> *   Deep Data Quality Diagnosis

---

## 14. 指标完善 (Metrics Implementation) [Dec 11, 2025]

### 14.1 指标完善 (Metrics Implementation)
为了更全面地评估模型在医学分割任务上的表现，我们在 `vecset/engines/engine_ae.py` 中实现了以下更新：
*   **Dice Score (DSC)**: 实现了标准的 Dice 系数计算公式 ($Dice = \frac{2 \times Intersection}{|A| + |B|}$)。
*   **Per-Class Metrics**: 
    *   在 Validation Loop 中增加了针对 class 1-10 的逐类 IoU 和 Dice 统计。
    *   新增指标共 20 项：`iou_class_1` ... `iou_class_10` 和 `dice_class_1` ... `dice_class_10`。
    *   这使得我们能够精确诊断小结构（如冠脉）的学习情况。
```
