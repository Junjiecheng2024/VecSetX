# Phase 3 训练与验证逻辑梳理

## 📊 核心策略

**从原来的**: 单一数据集内subset划分
**改为现在的**: 双数据集联合训练（Union of Subsets）

---

## 🗂️ 数据集配置

### Dataset A: Objaverse (基础集)
- **路径**: `/scratch/project_2016517/junjie/dataset/repaired_npz`
- **类别数**: 10类
- **类别**: `[1-10]` Myo, LA, LV, RA, RV, Ao, PA, LAA, Cor, PV
- **样本数**: ~798 (train)

### Dataset B: Dryad (扩展集)
- **路径**: `/scratch/project_2016517/junjie/dataset/dryad_npz`
- **类别数**: 16类
- **类别**: 
  - `[1-10]`: 与Dataset A重叠的10类
  - `[11-16]`: **新增6类** - SVC, IVC, RVW, LAW, CS, (1个预留)
- **样本数**: 22

### 合并索引
- **CSV文件**: `objaverse_train_combined.csv`
- **生成脚本**: `data_preparation/create_combined_csv.py`

---

## 🏋️ 训练逻辑 (phase3_structural/train.py)

### 1. 数据加载

```python
# 第149-160行
dataset_train = Objaverse(
    split='train_combined',  # 使用合并的CSV
    num_classes=16,          # 统一到16类
    partial_prob=0.8,        # ❌ 实际上不使用！
    min_remove=0,            # Phase 3标识
    max_remove=0             # Phase 3标识
)
```

**关键**：
- `split='train_combined'` → 加载 `objaverse_train_combined.csv`
- CSV包含两个目录的文件：`repaired_npz/` 和 `dryad_npz/`
- `num_classes=16` → 所有样本统一填充到16维

### 2. Batch组成

一个Batch可能包含：
```python
Batch 示例 (batch_size=2):
  样本1: 来自Dataset A (10类) → [1,2,3,4,5,6,7,8,9,10,0,0,0,0,0,0]
  样本2: 来自Dataset B (16类) → [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
```

### 3. Zero-Padding机制 (objaverse.py 第131-136行)

```python
if orig_cols < self.num_classes:
    # Dataset A (10类) 填充到 16类
    padding = torch.zeros((N, self.num_classes - orig_cols), dtype=torch.float32)
    surface_labels_tensor = torch.cat([surface_labels_tensor, padding], dim=1)
```

### 4. Valid Class Mask (objaverse.py 第395-400行)

```python
valid_class_mask = torch.ones(16, dtype=torch.float32)

if orig_cols < self.num_classes:
    # Dataset A: 类别11-16标记为无效
    valid_class_mask[orig_cols:] = 0.0  # [1,1,...,1,1,0,0,0,0,0,0]
```

**作用**：
- Dataset A样本: `[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0]`
- Dataset B样本: `[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]`

### 5. Masked Loss (engine_ae.py 第129-138行)

```python
# 分类损失计算
loss_cls_raw = F.binary_cross_entropy_with_logits(
    pred_cls_logits, target_cls_onehot, reduction='none'
)

# 应用mask：只在样本确实有该类标签时计算loss
mask_expanded = class_mask.unsqueeze(1).expand_as(loss_cls_raw)
loss_cls = (loss_cls_raw * mask_expanded).sum() / mask_expanded.sum()
```

**效果**：
- Dataset A样本预测类别11-16不会被惩罚（mask=0）
- Dataset B样本所有16类都正常计算loss
- **避免了知识遗忘**

---

## 🧪 验证逻辑

### 当前配置 (第161-169行)

```python
dataset_val = Objaverse(
    split='val',             # 使用普通val CSV
    num_classes=16,          # 统一16类
    # ❌ 没有partial_prob参数！
)
```

**问题分析**：

1. **验证集来源**: `objaverse_val.csv`
   - 只包含Dataset A的验证集（10类）
   - 不包含Dataset B（Dryad只有22个样本，可能全在训练集）

2. **验证策略**: 完整输入（无partial masking）
   - 每个验证样本都是完整的10类输入
   - 测试的是：**10类重建能力**
   - **不测试**：新增6类的学习效果

3. **CSV路径问题**：
   ```
   CSV格式: ,403.nii.img,dummy_category
   代码期望: category,filename,label
   实际路径: /dataset//403.nii.img.npz (双斜杠)
   ```
   已通过我的修复解决

---

## 🎯 关键发现

### ✅ 训练时发生了什么

```
Epoch循环:
  Batch 1: [DatasetA样本1(10类), DatasetA样本2(10类)]
  Batch 2: [DatasetB样本1(16类), DatasetB样本2(16类)]
  Batch 3: [DatasetA样本3(10类), DatasetB样本3(16类)] ← 混合！
  ...
```

**模型学到**：
- 从Dataset A: 10个基础心脏结构的几何形状
- 从Dataset B: 
  - 10个重叠类的**另一种分布**（Dryad vs Objaverse）
  - 6个新类的几何形状（SVC, IVC等）
  - 新类与旧类的**拓扑关系**

### ⚠️ 验证时没测试什么

- ❌ 新增6类(11-16)的重建质量
- ❌ 跨数据集泛化能力
- ❌ Dataset B样本的性能

### 🔧 验证集应该包含什么

**建议策略**：
创建验证集时应该包含：
1. Dataset A的val样本（测试10类重建）
2. Dataset B的部分样本（测试16类重建）

**创建方法**：
```bash
# 生成包含两个数据集的验证CSV
python create_combined_csv.py --include_val_b
```

---

## 📌 总结

### Phase 3的真实训练逻辑

| 方面 | 实际情况 |
|------|---------|
| **训练数据** | Dataset A (10类, 798) + Dataset B (16类, 22) |
| **训练策略** | 混合Batch + Zero-Padding + Masked Loss |
| **Partial Masking** | ❌ 不使用（`partial_prob=0.8`但`max_remove=0`） |
| **验证数据** | 只有Dataset A的val (10类) |
| **验证策略** | 完整输入，完整重建 |
| **核心创新** | 异构数据集联合训练，增量学习 |

### 当前存在的问题

1. ✅ **CSV路径问题** - 已修复
2. ⚠️ **验证集不包含Dataset B** - 需要添加
3. ⚠️ **没有测试新增6类的性能** - 需要per-class评估

### 下一步建议

1. **修复CSV路径** → 已完成
2. **创建完整验证集**：包含Dataset A + Dataset B的验证样本
3. **添加16类的per-class评估**：特别关注类别11-16
4. **监控Masked Loss**：确保Dataset A不会"干扰"Dataset B的学习

---

**Phase 3的本质**：
> 这不是"从subset推断全集"，而是"从两个不同分布的数据集中学习统一的16类心血管表示"。

这是一个**增量学习（Incremental Learning）**问题，而非补全问题！
