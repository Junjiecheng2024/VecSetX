
import sys
import os
import torch
import traceback

# Add the current directory to sys.path to mimic the training environment
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

print(f"Current Directory: {current_dir}")
print(f"Parent Directory (added to sys.path): {parent_dir}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not Set')}")

print("\n--- Step 1: Attempting Imports ---")
try:
    print("Importing vecset.utils.objaverse...")
    from vecset.utils.objaverse import Objaverse
    print("✅ Success: Imported Objaverse")
    
    print("Importing vecset.models.autoencoder...")
    from vecset.models import autoencoder
    print("✅ Success: Imported autoencoder")
    
    print("Importing vecset.engines.engine_ae...")
    from vecset.engines.engine_ae import train_one_epoch
    print("✅ Success: Imported engine_ae")

except ImportError as e:
    print(f"❌ FAIL: Import Error: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"❌ FAIL: User Initialization Error during imports: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n--- Step 2: Attempting Objaverse Initialization ---")
try:
    data_path = "/scratch/project_2016517/junjie/dataset/repaired_npz"
    print(f"Initializing Objaverse with data_path='{data_path}'...")
    
    # Check if data path exists
    if not os.path.exists(data_path):
        print(f"⚠️  WARNING: Data path {data_path} does not exist!")
    else:
        print(f"Data path exists.")

    dataset_train = Objaverse(
        split='train', 
        sdf_sampling=True, 
        sdf_size=4096, 
        surface_sampling=True, 
        surface_size=8192, 
        dataset_folder=data_path
    )
    print(f"✅ Success: Objaverse Initialized. Len: {len(dataset_train)}")
    
    # Try getting one item
    print("Attempting to get item 0...")
    item = dataset_train[0]
    print("✅ Success: Got item 0")

except Exception as e:
    print(f"❌ FAIL: Objaverse Initialization Error: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n--- Step 3: Attempting Model Initialization ---")
try:
    input_dim = 13
    print(f"Initializing Model 'learnable_vec1024x16_dim1024_depth24_nb' with input_dim={input_dim}...")
    model = autoencoder.__dict__['learnable_vec1024x16_dim1024_depth24_nb'](pc_size=8192, input_dim=input_dim)
    print("✅ Success: Model Initialized")
except Exception as e:
    print(f"❌ FAIL: Model Initialization Error: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 ALL CHECKS PASSED 🎉")
