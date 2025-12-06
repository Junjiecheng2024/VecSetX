import numpy as np
import glob
import os
import argparse

def check_file(npz_path):
    print(f"Checking {os.path.basename(npz_path)}...")
    try:
        data = np.load(npz_path)
        vol_sdf = data['vol_sdf'].flatten()  # Ensure 1D
        near_sdf = data['near_sdf'].flatten()  # Ensure 1D
        
        # 1. SDF Balance
        vol_pos = (vol_sdf > 0).mean()
        near_pos = (near_sdf > 0).mean()
        print(f"  [SDF Balance] Vol Pos: {vol_pos*100:.1f}% | Near Pos: {near_pos*100:.1f}%")
        
        # 2. Label Existence & Distribution
        if 'vol_labels' in data and 'near_labels' in data:
            vol_labels = data['vol_labels'].flatten()  # Ensure 1D
            near_labels = data['near_labels'].flatten()  # Ensure 1D
            
            print(f"  [Labels] Found vol_labels & near_labels")
            
            # Count classes
            unique, counts = np.unique(vol_labels, return_counts=True)
            print(f"    Vol Classes found: {len(unique)-1} (+Bg)")
            
            # 3. Consistency Check (The "Gold Standard")
            # Logic: If Inside (SDF < 0), should have Label > 0.
            #        If Outside (SDF >= 0), should have Label == 0.
            
            # Check 1: Inside points that have Background label (Missed Class?)
            # Use SDF < 0 to match the actual label assignment logic
            inside_but_bg = np.logical_and(vol_sdf < 0, vol_labels == 0).mean()
            
            # Check 2: Outside points that have Class label (Leaked Label?)
            # Use SDF >= 0 to match the actual label assignment logic
            outside_but_cls = np.logical_and(vol_sdf >= 0, vol_labels > 0).mean()
            
            consistency = 1.0 - (inside_but_bg + outside_but_cls)
            print(f"  [Consistency] {consistency*100:.2f}%")
            print(f"    - Inside but Bg (Missing Label): {inside_but_bg*100:.2f}% (Should be low, <5%)")
            print(f"    - Outside but Cls (Label Leak): {outside_but_cls*100:.2f}% (Should be ~0%)")
            
            if inside_but_bg > 0.1:
                print("    ⚠️ WARNING: High rate of unlabeled inside points. Labels might be too strict.")
            if outside_but_cls > 0.05:
                print("    ⚠️ WARNING: Labels leaking outside geometry.")
                
        else:
            print("  ⚠️ NO LABELS FOUND (Legacy Format)")
            
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to .npz file or directory')
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        files =  sorted(glob.glob(os.path.join(args.path, "*.npz")))
        print(f"Found {len(files)} files in {args.path}")
        # Check first 5 and random 5
        to_check = files[:5]
        if len(files) > 10:
            import random
            to_check += random.sample(files[5:], 5)
            
        for f in to_check:
            check_file(f)
    else:
        check_file(args.path)

if __name__ == "__main__":
    main()
