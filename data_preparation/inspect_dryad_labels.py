import numpy as np
import glob
import os
import argparse
import nibabel as nib

def analyze_npz(files):
    print(f"\nAnalyzing {len(files)} NPZ files...")
    all_labels = set()
    
    for i, f in enumerate(files):
        try:
            data = np.load(f)
            if 'vol_labels' in data:
                labels = data['vol_labels']
                unique = np.unique(labels)
                unique_nonzero = unique[unique != 0]
                
                # Update global set
                all_labels.update(unique_nonzero)
                
                # Print details for first 5 files
                if i < 5:
                    print(f"  {os.path.basename(f)}: Found {len(unique_nonzero)} classes -> {unique_nonzero}")
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    return sorted(list(all_labels))

def main():
    parser = argparse.ArgumentParser(description="Inspect Dryad Labels")
    parser.add_argument('dir', type=str, help='Directory containing .npz files')
    args = parser.parse_args()
    
    # 1. Check NPZ files
    npz_pattern = os.path.join(args.dir, "*.npz")
    npz_files = sorted(glob.glob(npz_pattern))
    
    if not npz_files:
        print(f"No .npz files found in {args.dir}")
        return

    print("="*60)
    print(f"INSPECTING LABELS in {args.dir}")
    print("="*60)
    
    found_labels = analyze_npz(npz_files)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total Unique Label IDs found across dataset: {found_labels}")
    print(f"Count: {len(found_labels)}")
    
    # Heuristic Interpretation
    print("\nPotential ID Mapping Interpretation:")
    print("  IDs 1-10 : Likely Overlapping Base Classes (from Dataset A definition)")
    print("  IDs 11+  : Likely New Dryad Classes")

if __name__ == "__main__":
    main()
