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
    *   **资源配置**: 4x A100 GPU, 32 CPUs/task, 24小时运行时限。
    *   **环境设置**: 设置 `PYTHONPATH` 和 `OMP_NUM_THREADS`。
    *   **路径配置**: 更新为集群上的实际路径 (`/projappl/...` 和 `/scratch/...`)。
    *   **WandB**: 启用 `--wandb` 标志。

### 决策记录
*   **Batch Size**: 确认在 A100 40GB 上使用 `Batch Size 64` 是安全的（显存占用约 26GB）。`Batch Size 128` 可能会导致 OOM（估算约 46GB），建议通过 `--accum_iter 2` 来模拟。
*   **框架选择**: 决定保持 **Native PyTorch** 架构，暂不迁移到 Lightning，以保持代码透明度和灵活性。
*   **配置管理**: 决定在 Phase 1 跑通后再引入 YAML/Hydra 配置文件，目前继续使用 `argparse`。
