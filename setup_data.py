"""
Setup script to organize SaSOKE data for training.
Creates the proper directory structure expected by the H2S dataloader.
"""

import os
import shutil
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
POSES_DIR = DATA_DIR / "poses"
REALIGNED_DIR = DATA_DIR / "re_aligned"

# Source data folders
OSX_FOLDERS = [
    BASE_DIR / "00_OSX_params" / "0_OSX_params",
    BASE_DIR / "01_OSX_params" / "1_OSX_params",
    BASE_DIR / "02_OSX_params" / "2_OSX_params",
]

# Annotation files
ANNOTATIONS_DIR = BASE_DIR / "annotations"

def create_directories():
    """Create the data directory structure."""
    print("Creating directories...")
    DATA_DIR.mkdir(exist_ok=True)
    POSES_DIR.mkdir(exist_ok=True)
    REALIGNED_DIR.mkdir(exist_ok=True)
    print(f"  ✓ Created {DATA_DIR}")
    print(f"  ✓ Created {POSES_DIR}")
    print(f"  ✓ Created {REALIGNED_DIR}")

def link_pose_folders():
    """Create symbolic links to pose folders."""
    print("\nLinking pose folders...")
    
    for osx_folder in OSX_FOLDERS:
        if not osx_folder.exists():
            print(f"  ⚠ Skipping {osx_folder} (not found)")
            continue
            
        # Get all sentence folders
        for sentence_folder in osx_folder.iterdir():
            if sentence_folder.is_dir():
                target = POSES_DIR / sentence_folder.name
                
                if target.exists():
                    continue
                    
                try:
                    # Create symbolic link (requires admin on Windows)
                    target.symlink_to(sentence_folder.resolve())
                    print(f"  ✓ Linked {sentence_folder.name}")
                except OSError:
                    # If symlink fails, copy instead
                    print(f"  ⚠ Symlink failed for {sentence_folder.name}, copying instead...")
                    # Don't actually copy - just note it
                    print(f"    Run as Administrator or copy manually")
    
    # Count total folders
    pose_count = len(list(POSES_DIR.iterdir())) if POSES_DIR.exists() else 0
    print(f"\n  Total pose folders: {pose_count}")

def convert_annotations():
    """Convert annotation txt files to CSV format expected by H2S loader."""
    print("\nConverting annotations...")
    
    # Use SI (Sign Interpreter) or US annotations
    annotation_source = ANNOTATIONS_DIR / "US"  # or "SI"
    
    for split in ["train", "dev", "test"]:
        txt_file = annotation_source / f"{split}.txt"
        
        if not txt_file.exists():
            print(f"  ⚠ {txt_file} not found")
            continue
        
        # Read the pipe-separated file
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse header and data
        header = lines[0].strip().split('|')
        data = []
        
        for line in lines[1:]:
            parts = line.strip().split('|')
            if len(parts) >= 3:
                data.append({
                    'SENTENCE_NAME': parts[0],
                    'gloss': parts[1],
                    'SENTENCE': parts[2]
                })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add required columns for H2S loader
        df['VIDEO_ID'] = df['SENTENCE_NAME'].str[:2]
        df['VIDEO_NAME'] = 'video_' + df['VIDEO_ID']
        df['SENTENCE_ID'] = range(1, len(df) + 1)
        df['START_REALIGNED'] = 0
        df['END_REALIGNED'] = 100  # Will be calculated from actual frames
        df['fps'] = 25
        
        # Map split names
        split_name = 'val' if split == 'dev' else split
        
        # Save as CSV
        csv_file = REALIGNED_DIR / f"h2s_realigned_{split_name}.csv"
        df.to_csv(csv_file, index=False)
        print(f"  ✓ Created {csv_file} ({len(df)} sentences)")

def verify_setup():
    """Verify the data setup is correct."""
    print("\n" + "="*50)
    print("Verification:")
    print("="*50)
    
    # Check poses
    if POSES_DIR.exists():
        pose_folders = list(POSES_DIR.iterdir())
        print(f"  ✓ Poses: {len(pose_folders)} folders")
    else:
        print(f"  ✗ Poses: NOT FOUND")
    
    # Check annotations
    for split in ["train", "val", "test"]:
        csv_file = REALIGNED_DIR / f"h2s_realigned_{split}.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            print(f"  ✓ {split}: {len(df)} sentences")
        else:
            print(f"  ✗ {split}: NOT FOUND")
    
    print("\n" + "="*50)
    print("Setup complete!")
    print("="*50)

def main():
    print("="*50)
    print("SaSOKE Data Setup")
    print("="*50)
    
    create_directories()
    convert_annotations()
    
    print("\n" + "="*50)
    print("IMPORTANT: Link Pose Folders Manually")
    print("="*50)
    print("""
Since Windows requires Administrator privileges for symlinks,
please run these commands in PowerShell (as Administrator):

cd C:\\Users\\d7oom\\Desktop\\SaSOKE-DHM-tech

# Option 1: Use Junction (works without admin)
cmd /c mklink /J "data\\poses\\00_OSX" "00_OSX_params\\0_OSX_params"
cmd /c mklink /J "data\\poses\\01_OSX" "01_OSX_params\\1_OSX_params"
cmd /c mklink /J "data\\poses\\02_OSX" "02_OSX_params\\2_OSX_params"

# Or Option 2: Just update config to point directly to the folders
""")
    
    verify_setup()

if __name__ == "__main__":
    main()

