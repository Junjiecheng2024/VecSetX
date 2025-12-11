# 数据准备记录 (Data Preparation Record)

本文档专门记录了 VecSetX 项目中所有数据集的准备过程、技术决策、遇到的问题及最终解决方案。

## 1. Dataset A: 原始心脏数据集 (10 Classes)

### 1.1 数据集概览
*   **来源**: 原始 Objaverse 类心脏 CT 数据集。
*   **样本量**: 998 个完整样本。
*   **格式**: `.nii.gz` 医学分割掩码 (Segmentation Mask)。
*   **分辨率**: 256x256x256 (各向同性)。
*   **类别**: 10 个解剖结构 + 背景。

### 1.2 预处理目标
将 NII 掩码转换为 VecSetX 模型所需的 `.npz` 格式，包含：
1.  **Surface Points**: 表面点云 (带 10 类标签的 One-hot 编码)。
2.  **Volume Points**: 体积内的随机采样点 (用于 SDF 监督)。
3.  **Near-Surface Points**: 靠近表面的采样点 (用于捕捉细节)。
4.  **SDF**: 每个点的符号距离函数值 (GT)。

### 1.3 准备历程与技术迭代

在 Phase 1 阶段，数据准备经历了一系列严重的质量问题和性能瓶颈，最终采用了实用主义的 "Lite" 方案。

#### 阶段一：初始尝试
*   **脚本**: `prepare_data.py` (v1)
*   **方法**: 
    *   使用 `skimage` 提取 Mesh。
    *   使用 `trimesh` 计算法向量近似 SDF。
*   **结果**: **失败**。训练初期 IoU 始终为 0。
*   **诊断**: 发现生成的 SDF 值严重偏移 (全为负值，均值 -89)，导致模型无法学习正确的零等势面。

#### 阶段二：算法修复与环境困境
*   **SDF 计算修复**: 尝试使用 `trimesh.contains()` (基于 Ray Casting) 进行精确的内外判断。
*   **依赖问题**: 该方法依赖 `rtree` 库，但在集群 Singularity 容器中缺少系统级依赖 (`libspatialindex`)，导致无法运行。
*   **环境修复**: 用户手动安装了 `rtree` 并加载了正确的 Python 模块，解决了运行环境问题。

#### 阶段三：Rtree 性能灾难 ("Perfect" 方案)
*   **脚本**: `prepare_data_perfect.py`
*   **尝试**: 使用精确的 Ray Casting 算法生成数据。
*   **结果**: 
    *   单个文件耗时 > 8 小时。
    *   内存占用 > 170 GB。
    *   预估完成时间需要 **333 天**。
*   **原因**: 心脏 Mesh 极其复杂 (数万面片)，与 20万个查询点进行 Ray Intersection 计算量过大。

#### 阶段四：最终方案 ("Lite" 方案)
*   **决策**: **Done is better than perfect**。放弃完美的 Ray Casting，转向启发式算法。
*   **脚本**: `prepare_data_lite.py`
*   **核心逻辑**: 
    *   使用距离质心的启发式规则 (`dist_to_center`) 来判断点在 Mesh 内部还是外部。
    *   `signs = np.where(dist_to_center < dists * 1.5, -1.0, 1.0)`
*   **效果**:
    *   **速度**: < 1 分钟/文件 (比 Perfect 版快 5300 倍)。
    *   **质量**: `vol_sdf` 均值回归到 0 附近，虽然 `near_sdf` 判断不如 Ray Casting 精确，但足以支持模型训练 (IoU > 0.3)。
*   **最终数据**: 已全部生成于 `/scratch/project_2016517/junjie/dataset/repaired_npz`。

---

## 2. Dataset B: Dryad 心脏数据集 (16 Classes)

### 2.1 数据集来源与特征
*   **来源**: Dryad 仓库 ("Automatic segmentation of multiple cardiovascular structures...")。
*   **样本量**: 22 个样本 (作为 Phase 3 的补充学习材料)。
*   **原始格式**: 2D PNG 切片 (Side-by-side: 左图 CT，右图 Mask)。
*   **新类别 (6类)**: 上腔静脉 (SVC), 下腔静脉 (IVC), 右室壁 (RVW), 左房壁 (LAW), 冠状窦 (CS) + 肺动脉 (PA)。
*   **总类别数**: 16 (包含 Dataset A 的 10 个重叠类别)。

### 2.2 准备与集成流程

#### 步骤 1: 格式转换 (Conversion)
*   **脚本**: `convert_dryad_v2.py`
*   **功能**:
    *   读取 PNG 图像，分割左右半部。
    *   解析文件名映射到 3D 空间。
    *   将 Mask 颜色映射到新的 Class IDs (11-16)。
    *   保存为 NII 格式 (`label_*.nii.gz`)。

#### 步骤 2: 预处理 (Preprocessing)
*   **脚本**: `prepare_data.py` (更新版)
*   **改进**: 
    *   增加了 `--pattern` 参数，支持筛选特定文件。
    *   更新了逻辑以支持 16 类的 One-hot 编码。

#### 步骤 3: 联合索引 (Indexing)
*   **脚本**: `create_combined_csv.py`
*   **功能**: 扫描 `repaired_npz` (Dataset A) 和 `dryad_npz` (Dataset B)，生成统一的 `objaverse_train_combined.csv`，使模型可以同时训练两个数据集。

---

## 3. 数据加载与动态兼容 (Data Loading & Compatibility)

为了在 Phase 3 中同时支持 10 类和 16 类数据，我们在数据加载层做了特殊设计。

### 3.1 动态类别支持
*   **文件**: `vecset/utils/objaverse.py`
*   **`num_classes` 参数**: 允许指定期望的输出类别数 (Phase 3 设为 16)。

### 3.2 自动零填充 (Zero-Padding)
*   **逻辑**: 当 Loader 读取到 Dataset A 的样本 (只有 10 类标签) 时，会自动在后面填充 6 个全零通道，使其形状变为 `(N, 19)` (3坐标 + 16类)，从而能与 Dataset B 的样本组成同一个 Batch。

### 3.3 标签冲突缓解 (Label Conflict Mitigation)
*   **问题**: Dataset A 没有标注 SVC，会被视为背景。这会抑制模型对 SVC 的学习。
*   **解决 (Loss Masking)**:
    *   Loader 返回 `valid_class_mask`。
    *   对于 Dataset A 样本，Mask 将 11-16 类置为无效。
    *   训练 Loss 计算时，忽略无效类别的预测误差。

---

## 4. Phase 1 数据生成方案的后续迭代 (Subsequent Data Generation Iterations)

> **注意**: 本章节记录了 Phase 1 中后期 (Dec 9 - Dec 11) 为了解决 Heuristic 算法缺陷而进行的一系列重大数据生成重构。

### 4.1 失败分析与数据生成修正 (Failure Analysis & Data Generation Pivot) [Dec 9, 2025]

#### 症状与现象
- **训练指标误导**: 模型训练 IoU 指标达到 ~0.70，但可视化结果呈 "Blob" 状。
- **根因**: 早期使用的 "Depth Ratio" 启发式算法在复杂心脏结构上失效，将大量背景空气误判为内部，导致模型学习了错误的标签。

#### 解决方案 (The Fix)
- **放弃 Heuristic**: 彻底废弃启发式算法。
- **全面采用 MeshToSDF**: 所有点（Volume & Near）均使用基于 Ray Casting 的精确算法计算 SDF。

### 4.2 Headless Cluster Fix - Trimesh-Based SDF [Dec 9, 2025]

#### 问题: No Display
`mesh_to_sdf` 库依赖 OpenGL/pyrender，需要 X11 显示，无法在 Headless 集群上运行。

#### 解决方案: Pure Trimesh
- 使用 `trimesh.proximity.signed_distance` 替换 `mesh_to_sdf`。
- 实现了 `compute_trimesh_sdf_batched`，支持 Headless 环境下的精确 SDF 计算。
- **性能**: 单文件 2-5 分钟，支持 batch processing。

### 4.3 量产方案: Fast Voxel-based Generation [Dec 9, 2025]

#### 核心算法重构：Voxel-based EDT
为了解决 Mesh Ray-casting 速度慢的问题，引入了基于体素的欧氏距离变换 (EDT)。
- **原理**: 直接对 3D Mask 进行 `scipy.ndimage.distance_transform_edt`，分别计算内部和外部距离场，组合得到 SDF。
- **性能**: 单文件处理时间降低到 **几秒钟**。
- **精度**: 三线性插值保证了连续空间的精度。

#### 采样策略优化：SDF Balance
- **Volume Points**: 50% Inside (Mask Jitter) + 50% Global (Random)。
- **Near Points**: 50% Push Out + 50% Push In (Based on Surface Normals)。
- **结果**: 实现了 45%-46% 的完美正负平衡。

### 4.4 数据采样策略重构 (Sampling Strategy Refactor) [Dec 11, 2025]

通过分析发现，原始“按面积加权”策略导致小结构（如冠脉）采样点极少。

#### "零和博弈"混合采样策略
1.  **Surface Points**: 引入 **Quota (低保)** + **Weighted (分红)** 机制。每个类别保底 2000 点，剩余按面积分配。冠脉点数提升 1000%。
2.  **Volume Points (Stratified)**: 内部点强制按类平分 (每类 2500 点)，外部点保持全局随机。

### 4.5 深入数据质量诊断 (Deep Data Quality Diagnosis) [Dec 11, 2025]

#### 全景分析 (998 样本)
- **心肌断裂**: 3.0% (30/998)。
- **类别丢失**: 0.1% (仅 Sample 75 丢失冠脉)。
- **整体质量**: 95% 样本完美。

#### 决策
- **拒绝 Soft-Fusion**: 为了 0.1% 的特例引入复杂算法不划算。
- **结论**: 现有数据质量完全足以支持 VecSetX 训练。
