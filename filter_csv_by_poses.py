"""
Filter CSV files to only include entries that have corresponding pose folders.
This fixes the FileNotFoundError during training.
"""
import pandas as pd
import os

def filter_csv(csv_path, poses_dir, output_path=None):
    """Filter CSV to only include rows with existing pose folders."""
    if output_path is None:
        output_path = csv_path
    
    print(f"\nProcessing: {csv_path}")
    df = pd.read_csv(csv_path)
    original_count = len(df)
    
    # Check which rows have corresponding pose folders
    valid_rows = []
    for idx, row in df.iterrows():
        sentence_name = row['SENTENCE_NAME']
        pose_folder = os.path.join(poses_dir, sentence_name)
        if os.path.exists(pose_folder):
            valid_rows.append(idx)
    
    # Filter dataframe
    df_filtered = df.loc[valid_rows].copy()
    filtered_count = len(df_filtered)
    
    print(f"  Original entries: {original_count}")
    print(f"  Valid entries (with pose folders): {filtered_count}")
    print(f"  Removed entries: {original_count - filtered_count}")
    
    # Save filtered CSV
    df_filtered.to_csv(output_path, index=False)
    print(f"  ✓ Saved to: {output_path}")
    
    return df_filtered

def main():
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, 'data')
    
    splits = ['train', 'val', 'test']
    
    print("=" * 60)
    print("Filtering CSV files to match existing pose folders")
    print("=" * 60)
    
    for split in splits:
        csv_path = os.path.join(data_dir, split, 're_aligned', f'how2sign_realigned_{split}_preprocessed_fps.csv')
        poses_dir = os.path.join(data_dir, split, 'poses')
        
        if os.path.exists(csv_path) and os.path.exists(poses_dir):
            filter_csv(csv_path, poses_dir)
        else:
            print(f"\n⚠ Skipping {split}: CSV or poses directory not found")
            print(f"  CSV: {csv_path} ({'exists' if os.path.exists(csv_path) else 'missing'})")
            print(f"  Poses: {poses_dir} ({'exists' if os.path.exists(poses_dir) else 'missing'})")
    
    print("\n" + "=" * 60)
    print("Done! Training should now work without FileNotFoundError.")
    print("=" * 60)

if __name__ == '__main__':
    main()

