# Phase 3: 结构共现 (子集并集)

**日期**: 2025-12-11
**目标**: 通过集成包含新结构的第二个数据集 (Dataset B)，扩展模型的解剖学知识库，实现"子集并集 (Union of Subsets)"的能力。

## 1. 数据集成策略

### 数据集构成
- **Dataset A (基础集)**: 原始类 Objaverse 心脏数据集 (10 个类别)。
- **Dataset B (Dryad)**: 来自 Dryad 仓库的 22 个样本 ("Automatic segmentation of multiple cardiovascular structures...")。
    - **原始格式**: 并排 PNG 图片。
    - **转换后**: 3D NII 体数据 (图像 + 掩码)。
    - **新类别**: 上腔静脉 (SVC), 下腔静脉 (IVC), 右室壁 (RVW), 左房壁 (LAW), 冠状窦 (CS)。
    - **总类别数**: 16 (10 个重叠类别 + 6 个新类别)。

### 流程更新
| 步骤 | 脚本 | 说明 |
| :--- | :--- | :--- |
| **1. 格式转换** | `convert_dryad_v2.py` | 解析 PNG，将文件夹映射为 Class ID 11-16，提取 CT/Mask 保存为 NII。 |
| **2. 预处理** | `prepare_data.py` | 增加了 `--pattern` 参数以支持 `label_*.nii.gz` 格式。生成 SDF/Surface 采样。 |
| **3. 索引构建** | `create_combined_csv.py` | 扫描 `repaired_npz` 和 `dryad_npz`，生成 `objaverse_train_combined.csv`。 |

## 2. 技术实现细节

### A. 动态类别支持与自动填充 (Padding)
为了在一个 Batch 中混合使用不同类别数 (10 vs 16) 的数据集：
- **`objaverse.py`**:
    - 增加了 `num_classes` 参数 (默认 10)。
    - 实现了 **零填充 (Zero-Padding)**: 如果加载的样本通道数少于 `num_classes`，则用 0 填充额外通道。

### B. 标签冲突缓解 (Loss Masking)
**问题**: Dataset A (10类) 将新结构 (SVC 等) 视为"背景" (Class 0)。如果直接训练，Dataset A 会惩罚模型对 SVC 的正确预测，导致模型"遗忘" Dataset B 学到的知识。
**解决方案**:
1.  **掩码生成**: `objaverse.py` 检测每个样本的原始列数，并返回 `valid_class_mask` (有效类别为 1.0，填充/未知类别为 0.0)。
2.  **Masked Loss**: `engine_ae.py` 将此掩码应用于 `BCEWithLogitsLoss`。
    ```python
    loss_cls = (loss_cls_raw * mask_expanded).sum() / mask_expanded.sum()
    ```
    *结果*: 模型只在样本确实包含某类别标签时，才计算该类别的 Loss。

### C. 向后兼容性
- 修复了共享引擎 (`engine_ae.py`)，如果未通过参数指定类别数，默认使用 10。
- 验证了 Phase 1 和 Phase 2 脚本无需修改即可正常运行。

## 3. 训练配置 (Phase 3)

**启动脚本**: `phase3_structural/run_phase3_sbatch.sh`

**关键超参数**:
- `input_dim`: 19 (3 坐标 + 16 类别) *注: 输入维度通常指表面特征维度*
- `nb_classes`: 16
- `train_split`: `train_combined`
- `data_path`: `/scratch/project_2016517/junjie/dataset` (父目录)

## 4. 当前状态
- 代码库已完全重构。
- 向后兼容性验证通过。
- 集群运行准备就绪:
    1. 运行 `prepare_data.py` 处理 Dryad 数据。
    2. 运行 `create_combined_csv.py` 生成索引。
    3. 提交任务 `sbatch phase3_structural/run_phase3_sbatch.sh`。
