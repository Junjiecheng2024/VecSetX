import os
import glob
import csv
import random

def create_csv(data_dir, output_dir, split_ratio=0.9):
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    random.shuffle(files)
    
    split_idx = int(len(files) * split_ratio)
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    
    # Objaverse CSV format: subdir, filename (without ext), ... (ignored)
    # My data is flat, so subdir is empty or '.'
    
    def write_csv(file_list, csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for file_path in file_list:
                filename = os.path.basename(file_path).replace('.npz', '')
                # subdir is empty string as files are directly in data_dir
                # But Objaverse joins npz_folder + row[0] + row[1] + '.npz'
                # So row[0] should be empty string
                writer.writerow(['', filename, 'dummy_category'])
                
    write_csv(train_files, os.path.join(output_dir, 'objaverse_train.csv'))
    write_csv(val_files, os.path.join(output_dir, 'objaverse_val.csv'))
    
    print(f"Created train.csv with {len(train_files)} files and val.csv with {len(val_files)} files.")

if __name__ == "__main__":
    create_csv('/home/user/persistent/vecset/data_npz', '/home/user/persistent/vecset/VecSetX/vecset/utils')
