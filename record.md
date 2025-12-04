# Phase 1 基础模型训练实施记录

本文档详细记录了 VecSetX 项目 Phase 1 (多类心脏结构重建) 的实施过程、文件修改原因及具体修改内容。

## 1. 数据准备 (Data Preparation)

### 文件: `prepare_data.py` (新增)
*   **原因**: 原始数据为 `.nii.gz` 格式的医学分割掩码，而 VecSetX 模型训练需要包含点云、SDF 值和表面法向量（可选）的 `.npz` 格式数据。此外，Phase 1 要求进行多类重建，因此需要在数据中包含类别信息。
*   **修改内容**:
    *   **网格提取**: 使用 `skimage.measure.marching_cubes` 从分割掩码中提取 3D 网格。
    *   **表面采样**: 实现了基于网格面积权重的表面点采样，确保采样点均匀分布。
    *   **类别编码**: 提取了 10 个解剖结构的类别标签，并对表面点进行了 **One-hot 编码**。生成的表面点数据维度为 `(N, 13)`，其中前 3 维为坐标，后 10 维为类别标签。
    *   **SDF 计算**: 使用 `trimesh` 和 `rtree` (KDTree) 计算体积点和近表面点的符号距离函数 (SDF) 值。为了提高速度，使用了法向量近似法。
    *   **归一化**: 将所有点坐标归一化到 `[-1, 1]` 范围，与模型输入要求一致。

### 文件: `create_csv.py` (新增)
*   **原因**: `Objaverse` 数据集加载器需要 CSV 文件来索引训练和验证数据。
*   **修改内容**: 编写脚本扫描数据目录，自动生成 `train.csv` 和 `val.csv`。

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
