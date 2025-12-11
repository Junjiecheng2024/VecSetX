import os
import glob
import csv
import random
import argparse

def create_combined_csv(dataset_root, output_dir, split_ratio=0.9):
    # dataset_root should be parent of 'repaired_npz' and 'dryad_npz'
    # e.g. /scratch/project_2016517/junjie/dataset/
    
    # 1. Dataset A: repaired_npz (10 classes)
    dir_a = 'repaired_npz'
    files_a = glob.glob(os.path.join(dataset_root, dir_a, "*.npz"))
    files_a = [os.path.join(dir_a, os.path.basename(f)) for f in files_a] # Store relative path including subdir
    
    # 2. Dataset B: dryad_npz (16 classes)
    dir_b = 'dryad_npz'
    files_b = glob.glob(os.path.join(dataset_root, dir_b, "*.npz"))
    files_b = [os.path.join(dir_b, os.path.basename(f)) for f in files_b]
    
    print(f"Found {len(files_a)} files in {dir_a}")
    print(f"Found {len(files_b)} files in {dir_b}")
    
    # Shuffle independently or together?
    # Better to shuffle together
    all_files = files_a + files_b
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * split_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    # Ensure validation set has at least some Dryad files if possible?
    # With random shuffle, likely yes.
    
    def write_csv(file_list, csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for rel_path in file_list:
                # rel_path is "subdir/filename.npz"
                subdir = os.path.dirname(rel_path)
                filename = os.path.basename(rel_path).replace('.npz', '')
                writer.writerow([subdir, filename, 'combined'])
                
    write_csv(train_files, os.path.join(output_dir, 'objaverse_train_combined.csv'))
    write_csv(val_files, os.path.join(output_dir, 'objaverse_val_combined.csv'))
    
    print(f"Created combined train.csv ({len(train_files)}) and val.csv ({len(val_files)}) in {output_dir}")

if __name__ == "__main__":
    create_combined_csv(
        '/scratch/project_2016517/junjie/dataset', 
        '/projappl/project_2016517/JunjieCheng/VecSetX/vecset/utils'
    )
