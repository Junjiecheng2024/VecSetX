#!/usr/bin/env python
"""
Analyze which classes are missing from vol_labels across the dataset
"""
import numpy as np
import glob
import os
from collections import Counter, defaultdict

def analyze_missing_classes(npz_dir):
    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    
    # Track missing classes
    missing_class_count = Counter()  # How many files miss each class
    files_by_num_classes = defaultdict(list)  # Group files by number of classes
    class_combinations = Counter()  # Track which combinations of classes are missing
    
    print(f"Analyzing {len(files)} files...")
    print("="*60)
    
    for f in files:
        data = np.load(f)
        vol_labels = data['vol_labels'].flatten()
        
        # Get unique classes (excluding background 0)
        present_classes = set([int(c) for c in np.unique(vol_labels) if c > 0])
        all_classes = set(range(1, 11))
        missing_classes = all_classes - present_classes
        
        num_classes = len(present_classes)
        files_by_num_classes[num_classes].append(os.path.basename(f))
        
        # Track missing classes
        for missing_cls in missing_classes:
            missing_class_count[missing_cls] += 1
        
        # Track missing combinations (for files with exactly 9 classes)
        if len(missing_classes) == 1:
            class_combinations[tuple(sorted(missing_classes))] += 1
    
    # Report
    print("\n📊 Missing Class Frequency (across all files)")
    print("="*60)
    class_names = {
        1: "Myocardium",
        2: "LA",
        3: "LV",
        4: "RA",
        5: "RV",
        6: "Aorta",
        7: "PA",
        8: "LAA",
        9: "Coronary",
        10: "PV"
    }
    
    for cls in range(1, 11):
        count = missing_class_count[cls]
        pct = count / len(files) * 100
        name = class_names.get(cls, f"Class {cls}")
        status = "❌" if pct > 50 else "⚠️" if pct > 20 else "✅"
        print(f"{status} Class {cls:2d} ({name:15s}): Missing in {count:4d} files ({pct:5.1f}%)")
    
    print("\n📋 Files by Number of Classes")
    print("="*60)
    for num_cls in sorted(files_by_num_classes.keys()):
        count = len(files_by_num_classes[num_cls])
        pct = count / len(files) * 100
        print(f"{num_cls} classes: {count:4d} files ({pct:5.1f}%)")
        
        # Show examples for files with <10 classes
        if num_cls < 10 and count <= 5:
            for fname in files_by_num_classes[num_cls][:5]:
                print(f"    - {fname}")
    
    print("\n🔍 For files with exactly 9 classes, which class is missing?")
    print("="*60)
    if class_combinations:
        for missing_combo, count in class_combinations.most_common():
            cls = missing_combo[0]
            name = class_names.get(cls, f"Class {cls}")
            pct = count / len(files_by_num_classes[9]) * 100 if 9 in files_by_num_classes else 0
            print(f"Missing Class {cls} ({name:15s}): {count:4d} files ({pct:5.1f}% of 9-class files)")
    
    print("\n" + "="*60)
    print(f"Total files analyzed: {len(files)}")
    print("="*60)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyze_missing_classes.py <npz_directory>")
        sys.exit(1)
    
    analyze_missing_classes(sys.argv[1])
