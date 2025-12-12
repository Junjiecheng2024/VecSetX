import numpy as np
import glob
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()
    
    files = sorted(glob.glob(os.path.join(args.path, "*.npz")))[:5] # Check first 5
    
    print(f"Checking first {len(files)} files in {args.path}")
    
    for f in files:
        data = np.load(f)
        if 'vol_labels' in data:
            labels = data['vol_labels']
            unique = np.unique(labels)
            print(f"{os.path.basename(f)}: Unique Labels = {unique}")
        else:
            print(f"{os.path.basename(f)}: No vol_labels found")

if __name__ == "__main__":
    main()
