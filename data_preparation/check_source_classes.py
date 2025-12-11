
import os
import glob
import nibabel as nib
import numpy as np
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Check class coverage in source .nii.gz files")
    parser.add_argument('input_dir', type=str, help='Directory containing .nii.gz files')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, '*.nii.gz')))
    print(f"Found {len(files)} files in {args.input_dir}")

    missing_class_files = []

    for f in tqdm(files):
        try:
            img = nib.load(f)
            data = img.get_fdata()
            unique_classes = np.unique(data)
            # Remove 0 (background)
            unique_classes = unique_classes[unique_classes != 0]
            
            if len(unique_classes) < 10:
                missing = set(range(1, 11)) - set(unique_classes)
                print(f"⚠️  {os.path.basename(f)} has only {len(unique_classes)} classes. Missing: {missing}")
                missing_class_files.append((os.path.basename(f), missing))
        except Exception as e:
            print(f"❌ Error reading {f}: {e}")

    print("\n" + "="*50)
    print(f"Summary: Found {len(missing_class_files)} files with missing classes.")
    if len(missing_class_files) > 0:
        for fname, missing in missing_class_files:
             print(f" - {fname}: Missing {missing}")
    else:
        print("✅ All files have complete 10 classes!")
    print("="*50)

if __name__ == '__main__':
    main()
