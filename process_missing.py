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
    print(f"\nMissing files:")
    for i, name in enumerate(missing[:20]):  # Show first 20
        idx = int(name.split('.')[0])
        print(f"  {i+1}. {name}.nii.gz (index {idx})")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")
    
    # Create temp directory with symlinks
    temp_dir = "/scratch/project_2016517/junjie/dataset/missing_files_temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"\n🔧 Creating temporary directory with symlinks...")
    for name in missing:
        src = f"{input_dir}/{name}.nii.gz"
        dst = f"{temp_dir}/{name}.nii.gz"
        if os.path.exists(src):
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)
    
    print(f"✅ Created {len(missing)} symlinks in {temp_dir}")
    
    # Recommended command
    cmd = f"""python prepare_data.py \\
    --input_dir {temp_dir} \\
    --output_dir {output_dir} \\
    --vol_threshold 0.85 \\
    --file_workers 16"""
    
    print(f"\n🚀 Now run this command to process ONLY the missing files:\n")
    print(cmd)
    print(f"\n⚠️  After completion, you can delete the temp directory:")
    print(f"rm -rf {temp_dir}")