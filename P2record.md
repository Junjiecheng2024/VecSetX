# Phase 2 补全任务训练实施记录 (Probabilistic Completion)

本文档详细记录了 VecSetX 项目 Phase 2 (概率补全/Probabilistic Completion) 的实施过程、架构调整及关键代码修改。

## 1. 项目架构重构 (Project Refactoring)

为了支持多阶段开发并保持项目整洁，对文件结构进行了全面重构。

### 行动记录
*   **创建目录结构**:
    *   `vecset/`: 核心库 (保留 Models, Engines, Utils)。
    *   `phase1_reconstruction/`: 存放 Phase 1 (全心重建) 相关的训练、推理脚本及 SLURM 脚本。
    *   `phase2_completion/`: 存放 Phase 2 (概率补全) 相关的开发脚本。
    *   `data_preparation/`: 存放所有数据预处理与质量检查脚本 (`prepare_data.py`, `check_data_quality.py` 等)。
*   **脚本迁移与修复**:
    *   将 `main_ae.py` 移动并重命名为 `phase1_reconstruction/train.py`。
    *   修改所有移动后脚本的 `import` 路径，添加 `sys.path.append('..')` 以确保能正确引用根目录下的 `vecset` 包。

## 2. 核心功能实现: 随机类别丢弃 (Random Class Dropout)

Phase 2 的核心目标是训练模型从残缺输入中推断完整结构。我们采用 **On-the-fly Data Augmentation** 策略，无需重新生成数据集。

### 文件: `vecset/utils/objaverse.py`
*   **修改内容**:
    *   在 `__init__` 中引入新参数: `partial_prob` (触发概率), `min_remove` (最少丢弃数), `max_remove` (最多丢弃数)。
    *   在 `__getitem__` 中实现 **Random Class Dropout Logic**:
        *   当处于 `train` 模式且 `rand() < partial_prob` 时触发。
        *   从当前样本存在的类别中，随机选择 $k$ (`[min, max]`) 个类别。
        *   **Input Masking**: 仅在 `surface` (输入编码器点云) 中移除这些类别的点。
        *   **Target Preservation**: `points` (查询点) 和 `sdf` (监督信号) 保持不变，代表完整解剖结构。
    *   **目的**: 强迫 AutoEncoder 学习解剖结构之间的共生关系（例如：看到左心室就应该“脑补”出相连的主动脉根部），从而实现概率补全。

## 3. 训练流程适配 (Training Pipeline)

### 文件: `phase2_completion/train.py`
*   **来源**: 基于 `phase1_reconstruction/train.py` 复制。
*   **修改内容**:
    *   添加命令行参数: `--partial_prob`, `--min_remove`, `--max_remove`。
    *   将这些参数传递给 `Objaverse` 数据加载器初始化函数。

### 文件: `phase2_completion/run_phase2_sbatch.sh`
*   **修改内容**:
    *   **Job Name**: `vecset_phase2`。
    *   **Output Dir**: `output/ae/phase2_completion`。
    *   **核心配置**:
        ```bash
        --partial_prob 0.8  # 80% 的概率触发 Masking，高强度迫使模型泛化
        --min_remove 2      # 至少移除 2 个器官
        --max_remove 5      # 最多移除 5 个器官 (半颗心)
        ```

## 4. 验证工具开发 (Verification Tool)

### 文件: `phase2_completion/infer_completion.py`
*   **功能**: 专门用于验证模型的补全能力。
*   **新增特性**:
    *   参数 `--drop_classes`: 允许用户在推理时手动指定要移除的类别 ID (如 `1,9`)。
    *   **可视化**: 生成对比图，展示 输入点云(残缺) vs 预测SDF(完整)。
    *   **预期效果**: 即使输入点云中缺少了冠脉(Class 9)，如果模型训练成功，预测的 SDF 重建结果中仍应包含冠脉结构。
