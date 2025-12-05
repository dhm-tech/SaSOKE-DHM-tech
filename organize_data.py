"""
Organize SaSOKE data into the structure expected by the H2S dataloader.
Creates symbolic links for poses and copies annotation files.
"""

import os
import shutil
import pandas as pd
from pathlib import Path

BASE_DIR = Path(".")

# Source pose folders (junctions created earlier)
POSE_SOURCES = {
    "00": BASE_DIR / "data" / "poses" / "00",
    "01": BASE_DIR / "data" / "poses" / "01", 
    "02": BASE_DIR / "data" / "poses" / "02",
}

def organize_split(split_name):
    """Organize data for one split (train/val/test)."""
    print(f"\nOrganizing {split_name}...")
    
    # Read annotation file
    csv_src = BASE_DIR / "data" / "re_aligned" / f"h2s_realigned_{split_name}.csv"
    if not csv_src.exists():
        print(f"  ✗ Annotation file not found: {csv_src}")
        return
    
    df = pd.read_csv(csv_src)
    
    # Create target annotation file with correct name
    target_dir = BASE_DIR / "data" / split_name / "re_aligned"
    target_csv = target_dir / f"how2sign_realigned_{split_name}_preprocessed_fps.csv"
    
    # Ensure fps column exists
    if 'fps' not in df.columns:
        df['fps'] = 25
    
    # Save with correct name
    df.to_csv(target_csv, index=False)
    print(f"  ✓ Created {target_csv}")
    
    # Link pose folders
    poses_dir = BASE_DIR / "data" / split_name / "poses"
    sentence_names = df['SENTENCE_NAME'].unique()
    
    linked = 0
    skipped = 0
    
    for name in sentence_names:
        # Determine source folder based on prefix (00, 01, 02)
        prefix = name[:2]
        
        if prefix in POSE_SOURCES:
            source = POSE_SOURCES[prefix] / name
        else:
            # Try all sources
            source = None
            for src_prefix, src_path in POSE_SOURCES.items():
                potential = src_path / name
                if potential.exists():
                    source = potential
                    break
        
        if source and source.exists():
            target = poses_dir / name
            if not target.exists():
                try:
                    # Use junction on Windows
                    os.system(f'cmd /c mklink /J "{target}" "{source.resolve()}"')
                    linked += 1
                except Exception as e:
                    print(f"  ✗ Failed to link {name}: {e}")
                    skipped += 1
        else:
            skipped += 1
    
    print(f"  ✓ Linked {linked} pose folders, skipped {skipped}")

def verify_setup():
    """Verify the data structure."""
    print("\n" + "="*50)
    print("Verification:")
    print("="*50)
    
    for split in ["train", "val", "test"]:
        print(f"\n{split.upper()}:")
        
        # Check annotation file
        csv_path = BASE_DIR / "data" / split / "re_aligned" / f"how2sign_realigned_{split}_preprocessed_fps.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"  ✓ Annotations: {len(df)} sentences")
        else:
            print(f"  ✗ Annotations: NOT FOUND")
        
        # Check pose folders
        poses_dir = BASE_DIR / "data" / split / "poses"
        if poses_dir.exists():
            pose_count = len(list(poses_dir.iterdir()))
            print(f"  ✓ Poses: {pose_count} folders")
        else:
            print(f"  ✗ Poses: NOT FOUND")

def main():
    print("="*50)
    print("Organizing SaSOKE Data")
    print("="*50)
    
    # Organize each split
    for split in ["train", "val", "test"]:
        organize_split(split)
    
    verify_setup()
    
    print("\n" + "="*50)
    print("Done!")
    print("="*50)

if __name__ == "__main__":
    main()

