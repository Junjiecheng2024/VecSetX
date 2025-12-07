import os
import glob
import subprocess
input_dir = "/scratch/project_2016517/junjie/dataset/repaired_shape"
output_dir = "/scratch/project_2016517/junjie/dataset/repaired_npz"
# Find missing files
input_files = glob.glob(f"{input_dir}/*.nii.gz")
input_basenames = {os.path.basename(f).replace('.nii.gz', '') for f in input_files}
output_files = glob.glob(f"{output_dir}/*.npz")
output_basenames = {os.path.basename(f).replace('.npz', '') for f in output_files}
missing = sorted(input_basenames - output_basenames, key=lambda x: int(x.split('.')[0]))
print(f"Found {len(missing)} missing files")
if len(missing) == 0:
    print("All files already processed!")
else:
    # Get indices
    indices = [int(name.split('.')[0]) for name in missing]
    min_idx = min(indices)
    max_idx = max(indices)
    
    print(f"Missing indices range: {min_idx} to {max_idx}")
    print(f"Total missing: {len(missing)}")
    
    # Process in batches
    cmd = f"""python prepare_data.py \\
    --input_dir {input_dir} \\
    --output_dir {output_dir} \\
    --vol_threshold 0.85 \\
    --file_workers 16 \\
    --start_idx {min_idx} \\
    --end_idx {max_idx + 1}"""
    
    print(f"\nRecommended command:\n{cmd}")