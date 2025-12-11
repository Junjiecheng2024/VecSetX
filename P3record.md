# Phase 3 结构共现训练实施记录 (Structural Co-occurrence)

本文档记录 VecSetX 项目 Phase 3 (Structural Co-occurrence/结构子集组合学习) 的实施细节。

## 1. 核心目标
Phase 3 的目标是让模型学会从**特定的解剖子集**中恢复完整心脏。这不同于 Phase 2 的随机修补，而是模拟真实的临床场景（例如，只有超声心动图显示的四腔心切面，或者只有造影显示的冠脉和血管）。

## 2. 核心实现: Subset Masking (子集掩码)
我们在 `vecset/utils/objaverse.py` 中实现了新的掩码逻辑。

### 定义的子集 (Subsets)
我们定义了以下具临床意义的子集：
1.  **Left Heart (左心系统)**: Myocardium(1), LA(2), LV(3), Aorta(6), LAA(8), PV(10)。
2.  **Right Heart (右心系统)**: RA(4), RV(5), PA(7)。
3.  **Four Chambers (四腔心)**: LA(2), LV(3), RA(4), RV(5)。
4.  **Great Vessels (大血管)**: Aorta(6), PA(7), Coronary(9)。

### 逻辑控制
通过以下参数组合触发 Phase 3 逻辑：
*   `max_remove = 0` (禁用 Phase 2 的随机移除)
*   `partial_prob > 0` (启用 Phase 3 的子集模式)

当触发时，数据加载器会**随机选择一个子集**，并**仅保留**属于该子集的输入点云，移除所有其他点。这强迫模型从子集推断整体（例如：Given Right Heart -> Predict Full Heart）。

## 3. 训练配置
### 文件: `phase3_structural/train.py`
复用 Phase 2 的训练脚本，支持参数传递。

### 文件: `phase3_structural/run_phase3_sbatch.sh`
*   **Job Name**: `vecset_phase3`
*   **Port**: `29502` (避免与 P1/P2 冲突)
*   **关键参数**:
    ```bash
    --partial_prob 0.8  # 80% 概率触发子集掩码
    --min_remove 0      # 设为 0 以激活 Objaverse 中的 Phase 3 逻辑
    --max_remove 0      # 设为 0 以激活 Objaverse 中的 Phase 3 逻辑
    ```

## 4. 预期成果
训练完成后，该模型应具备“解剖联想”能力：
*   输入：仅有右心房和右心室。
*   输出：完整的心脏（包括左心室、主动脉等），且位置和大小应符合解剖学上的共生关系。
