import numpy as np
import glob
import os
import argparse

def check_file(npz_path, verbose=False):
    """Check a single file and return statistics"""
    try:
        data = np.load(npz_path)
        vol_sdf = data['vol_sdf'].flatten()
        near_sdf = data['near_sdf'].flatten()
        
        # SDF Balance
        vol_pos = (vol_sdf > 0).mean()
        near_pos = (near_sdf > 0).mean()
        
        # Label check
        stats = {
            'vol_pos': vol_pos,
            'near_pos': near_pos,
            'consistency': 0,
            'num_classes': 0,
            'has_labels': False
        }
        
        if 'vol_labels' in data and 'near_labels' in data:
            vol_labels = data['vol_labels'].flatten()
            near_labels = data['near_labels'].flatten()
            
            stats['has_labels'] = True
            
            # Count classes
            unique = np.unique(vol_labels)
            stats['num_classes'] = len(unique) - 1  # Exclude background
            
            # Consistency
            inside_but_bg = np.logical_and(vol_sdf < 0, vol_labels == 0).mean()
            outside_but_cls = np.logical_and(vol_sdf >= 0, vol_labels > 0).mean()
            stats['consistency'] = 1.0 - (inside_but_bg + outside_but_cls)
            stats['inside_but_bg'] = inside_but_bg
            stats['outside_but_cls'] = outside_but_cls
        
        if verbose:
            print(f"✓ {os.path.basename(npz_path)}: Vol={vol_pos*100:.1f}%, Near={near_pos*100:.1f}%, Classes={stats['num_classes']}, Consistency={stats['consistency']*100:.1f}%")
        
        return stats
    except Exception as e:
        print(f"✗ Error reading {os.path.basename(npz_path)}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to .npz file or directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show per-file details')
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "*.npz")))
        print(f"Found {len(files)} files in {args.path}")
        print("="*60)
        
        # Collect statistics
        all_stats = []
        for f in files:
            stats = check_file(f, verbose=args.verbose)
            if stats:
                all_stats.append(stats)
        
        if not all_stats:
            print("No valid files found!")
            return
        
        # Aggregate statistics
        print("\n" + "="*60)
        print("AGGREGATE STATISTICS")
        print("="*60)
        
        vol_pos_vals = [s['vol_pos'] for s in all_stats]
        near_pos_vals = [s['near_pos'] for s in all_stats]
        consistency_vals = [s['consistency'] for s in all_stats if s['has_labels']]
        num_classes_vals = [s['num_classes'] for s in all_stats if s['has_labels']]
        
        print(f"\n📊 SDF Balance (Vol Positive %)")
        print(f"  Mean:   {np.mean(vol_pos_vals)*100:.1f}%")
        print(f"  Std:    {np.std(vol_pos_vals)*100:.1f}%")
        print(f"  Range:  [{np.min(vol_pos_vals)*100:.1f}%, {np.max(vol_pos_vals)*100:.1f}%]")
        
        print(f"\n📊 SDF Balance (Near Positive %)")
        print(f"  Mean:   {np.mean(near_pos_vals)*100:.1f}%")
        print(f"  Std:    {np.std(near_pos_vals)*100:.1f}%")
        print(f"  Range:  [{np.min(near_pos_vals)*100:.1f}%, {np.max(near_pos_vals)*100:.1f}%]")
        
        if consistency_vals:
            print(f"\n✅ Label Consistency")
            print(f"  Mean:   {np.mean(consistency_vals)*100:.1f}%")
            print(f"  Std:    {np.std(consistency_vals)*100:.1f}%")
            print(f"  Range:  [{np.min(consistency_vals)*100:.1f}%, {np.max(consistency_vals)*100:.1f}%]")
            
            # Class coverage distribution
            from collections import Counter
            class_dist = Counter(num_classes_vals)
            print(f"\n📋 Class Coverage Distribution")
            for n_classes in sorted(class_dist.keys()):
                count = class_dist[n_classes]
                pct = count / len(num_classes_vals) * 100
                print(f"  {n_classes} classes: {count} files ({pct:.1f}%)")
            
            # Overall assessment
            print(f"\n{'='*60}")
            avg_consistency = np.mean(consistency_vals)
            avg_vol_pos = np.mean(vol_pos_vals)
            avg_near_pos = np.mean(near_pos_vals)
            
            print("OVERALL ASSESSMENT")
            print(f"{'='*60}")
            
            if avg_consistency > 0.95:
                print("✅ Consistency: EXCELLENT (>95%)")
            elif avg_consistency > 0.90:
                print("✅ Consistency: GOOD (>90%)")
            else:
                print("⚠️  Consistency: NEEDS IMPROVEMENT (<90%)")
            
            if 45 <= avg_vol_pos <= 70:
                print("✅ Vol SDF Balance: GOOD (45-70%)")
            else:
                print("⚠️  Vol SDF Balance: Outside recommended range")
            
            if 48 <= avg_near_pos <= 54:
                print("✅ Near SDF Balance: EXCELLENT (48-54%)")
            elif 45 <= avg_near_pos <= 60:
                print("✅ Near SDF Balance: GOOD (45-60%)")
            else:
                print("⚠️  Near SDF Balance: Outside recommended range")
            
            all_have_10_classes = all(n == 10 for n in num_classes_vals)
            if all_have_10_classes:
                print("✅ Class Coverage: PERFECT (All files have 10 classes)")
            else:
                pct_with_10 = sum(1 for n in num_classes_vals if n == 10) / len(num_classes_vals) * 100
                print(f"⚠️  Class Coverage: {pct_with_10:.1f}% of files have all 10 classes")
        
        print(f"\n{'='*60}")
        print(f"Total files analyzed: {len(all_stats)}")
        print(f"{'='*60}")
        
    else:
        # Single file
        check_file(args.path, verbose=True)

if __name__ == "__main__":
    main()
