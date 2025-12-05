#!/usr/bin/env python3
"""
数据质量检查脚本
检查 npz 数据集的质量，包括 SDF 值分布、标签、坐标范围等
"""

import numpy as np
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

def check_single_sample(npz_path):
    """检查单个样本的数据质量"""
    try:
        with np.load(npz_path) as data:
            results = {}
            
            # 基本信息
            results['filename'] = os.path.basename(npz_path)
            results['keys'] = list(data.keys())
            
            # 检查各个数组
            if 'vol_points' in data:
                vol_points = data['vol_points']
                results['vol_points_shape'] = vol_points.shape
                results['vol_points_range'] = (vol_points.min(), vol_points.max())
                
            if 'vol_sdf' in data:
                vol_sdf = data['vol_sdf']
                results['vol_sdf_shape'] = vol_sdf.shape
                results['vol_sdf_range'] = (vol_sdf.min(), vol_sdf.max())
                results['vol_sdf_mean'] = vol_sdf.mean()
                results['vol_sdf_positive_ratio'] = (vol_sdf > 0).sum() / vol_sdf.size
                
            if 'near_points' in data:
                near_points = data['near_points']
                results['near_points_shape'] = near_points.shape
                results['near_points_range'] = (near_points.min(), near_points.max())
                
            if 'near_sdf' in data:
                near_sdf = data['near_sdf']
                results['near_sdf_shape'] = near_sdf.shape
                results['near_sdf_range'] = (near_sdf.min(), near_sdf.max())
                results['near_sdf_mean'] = near_sdf.mean()
                results['near_sdf_positive_ratio'] = (near_sdf > 0).sum() / near_sdf.size
                
            if 'surface_points' in data:
                surface = data['surface_points']
                results['surface_points_shape'] = surface.shape
                results['surface_points_range'] = (surface.min(), surface.max())
                
            if 'surface_labels' in data:
                labels = data['surface_labels']
                results['surface_labels_shape'] = labels.shape
                results['surface_labels_range'] = (labels.min(), labels.max())
                # 检查是否是 one-hot
                results['labels_sum_per_point'] = labels.sum(axis=1).mean()  # 应该都是 1
                results['num_classes'] = labels.shape[1] if len(labels.shape) > 1 else 1
                
            return results
    except Exception as e:
        return {'filename': os.path.basename(npz_path), 'error': str(e)}

def analyze_dataset(data_dir, num_samples=10):
    """分析整个数据集的前 N 个样本"""
    npz_files = sorted(Path(data_dir).glob('*.npz'))
    
    print(f"=" * 80)
    print(f"数据集路径: {data_dir}")
    print(f"总文件数: {len(npz_files)}")
    print(f"检查样本数: {min(num_samples, len(npz_files))}")
    print(f"=" * 80)
    
    all_results = []
    
    for i, npz_file in enumerate(npz_files[:num_samples]):
        print(f"\n{'='*80}")
        print(f"样本 {i+1}/{num_samples}: {npz_file.name}")
        print(f"{'='*80}")
        
        result = check_single_sample(npz_file)
        all_results.append(result)
        
        if 'error' in result:
            print(f"❌ 错误: {result['error']}")
            continue
            
        # 打印详细信息
        print(f"\n📦 数据键: {result.get('keys', [])}")
        
        print(f"\n📍 坐标范围:")
        print(f"  - vol_points: {result.get('vol_points_range', 'N/A')}")
        print(f"  - near_points: {result.get('near_points_range', 'N/A')}")
        print(f"  - surface_points: {result.get('surface_points_range', 'N/A')}")
        
        print(f"\n📏 SDF 统计:")
        print(f"  - vol_sdf 范围: {result.get('vol_sdf_range', 'N/A')}")
        print(f"  - vol_sdf 均值: {result.get('vol_sdf_mean', 'N/A'):.4f}")
        print(f"  - vol_sdf 正值比例: {result.get('vol_sdf_positive_ratio', 'N/A'):.2%}")
        print(f"  - near_sdf 范围: {result.get('near_sdf_range', 'N/A')}")
        print(f"  - near_sdf 均值: {result.get('near_sdf_mean', 'N/A'):.4f}")
        print(f"  - near_sdf 正值比例: {result.get('near_sdf_positive_ratio', 'N/A'):.2%}")
        
        print(f"\n🏷️ 标签信息:")
        print(f"  - 标签形状: {result.get('surface_labels_shape', 'N/A')}")
        print(f"  - 每点标签和: {result.get('labels_sum_per_point', 'N/A'):.4f} (应该 ≈ 1.0)")
        print(f"  - 类别数: {result.get('num_classes', 'N/A')}")
        
        # 检查潜在问题
        print(f"\n⚠️ 潜在问题检查:")
        issues = []
        
        # 检查坐标是否归一化到 [-1, 1]
        if result.get('vol_points_range'):
            vmin, vmax = result['vol_points_range']
            if vmin < -1.5 or vmax > 1.5:
                issues.append(f"vol_points 未归一化到 [-1,1]: {result['vol_points_range']}")
                
        # 检查 SDF 值是否合理
        if result.get('vol_sdf_range'):
            sdf_min, sdf_max = result['vol_sdf_range']
            if abs(sdf_min) > 100 or abs(sdf_max) > 100:
                issues.append(f"vol_sdf 范围异常大: {result['vol_sdf_range']}")
                
        # 检查标签和是否为 1
        if result.get('labels_sum_per_point'):
            label_sum = result['labels_sum_per_point']
            if abs(label_sum - 1.0) > 0.01:
                issues.append(f"标签和不为1: {label_sum:.4f} (可能不是 one-hot)")
                
        if issues:
            for issue in issues:
                print(f"  ⚠️ {issue}")
        else:
            print(f"  ✅ 未发现明显问题")
    
    # 统计总结
    print(f"\n{'='*80}")
    print(f"总结统计")
    print(f"{'='*80}")
    
    # 计算平均值
    avg_vol_sdf_mean = np.mean([r.get('vol_sdf_mean', 0) for r in all_results if 'vol_sdf_mean' in r])
    avg_near_sdf_mean = np.mean([r.get('near_sdf_mean', 0) for r in all_results if 'near_sdf_mean' in r])
    
    print(f"平均 vol_sdf: {avg_vol_sdf_mean:.4f}")
    print(f"平均 near_sdf: {avg_near_sdf_mean:.4f}")
    
    # ⚠️ 关键检查：SDF 均值应该接近 0
    if abs(avg_vol_sdf_mean) > 0.5:
        print(f"\n❌ 严重问题: vol_sdf 均值 {avg_vol_sdf_mean:.4f} 偏离 0 太多！")
        print(f"   这可能导致模型系统性偏移，IoU = 0")
    
    if abs(avg_near_sdf_mean) > 0.5:
        print(f"\n❌ 严重问题: near_sdf 均值 {avg_near_sdf_mean:.4f} 偏离 0 太多！")
        print(f"   这可能导致模型系统性偏移，IoU = 0")
    
    return all_results

def plot_sdf_distribution(data_dir, num_samples=5, output_path='sdf_distribution.png'):
    """绘制 SDF 分布直方图"""
    npz_files = sorted(Path(data_dir).glob('*.npz'))[:num_samples]
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    for i, npz_file in enumerate(npz_files):
        with np.load(npz_file) as data:
            vol_sdf = data['vol_sdf']
            near_sdf = data['near_sdf']
            
            # Vol SDF 分布
            axes[0, i].hist(vol_sdf.flatten(), bins=50, alpha=0.7, edgecolor='black')
            axes[0, i].axvline(x=0, color='r', linestyle='--', label='SDF=0')
            axes[0, i].set_title(f'{npz_file.stem}\nvol_sdf')
            axes[0, i].set_xlabel('SDF value')
            axes[0, i].set_ylabel('Count')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Near SDF 分布
            axes[1, i].hist(near_sdf.flatten(), bins=50, alpha=0.7, edgecolor='black', color='orange')
            axes[1, i].axvline(x=0, color='r', linestyle='--', label='SDF=0')
            axes[1, i].set_title(f'near_sdf')
            axes[1, i].set_xlabel('SDF value')
            axes[1, i].set_ylabel('Count')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 SDF 分布图已保存到: {output_path}")
    plt.close()

if __name__ == '__main__':
    data_dir = '/scratch/project_2016517/junjie/dataset/repaired_npz'
    
    # 分析数据集
    results = analyze_dataset(data_dir, num_samples=10)
    
    # 绘制分布图
    plot_sdf_distribution(data_dir, num_samples=5, output_path='/projappl/project_2016517/JunjieCheng/VecSetX/sdf_distribution.png')
    
    print(f"\n✅ 数据质量检查完成！")
