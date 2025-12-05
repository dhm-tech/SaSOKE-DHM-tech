"""
Generate motion tokens using the full model's VAEs.
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from pathlib import Path
import yaml

import sys
sys.path.insert(0, '.')

def load_pose_sample(pose_dir, name):
    """Load and process a single pose sample (matching load_data.py logic)."""
    base_dir = pose_dir / name
    if not base_dir.exists():
        return None
    
    all_files = sorted([f for f in os.listdir(base_dir) if f.endswith('.pkl')])
    if not all_files or len(all_files) < 4:
        return None
    
    # Build 179-dim pose vector for each frame
    clip_poses = np.zeros([len(all_files), 179])
    
    for i, f in enumerate(all_files):
        with open(base_dir / f, 'rb') as fp:
            data = pickle.load(fp)
        
        root = data.get('smplx_root_pose', np.zeros(3)).flatten()[:3]
        body = data.get('smplx_body_pose', np.zeros(63)).flatten()[:63]
        lhand = data.get('smplx_lhand_pose', np.zeros(45)).flatten()[:45]
        rhand = data.get('smplx_rhand_pose', np.zeros(45)).flatten()[:45]
        jaw = data.get('smplx_jaw_pose', np.zeros(3)).flatten()[:3]
        shape = data.get('smplx_shape', np.zeros(10)).flatten()[:10]
        expr = data.get('smplx_expr', np.zeros(10)).flatten()[:10]
        
        clip_poses[i] = np.concatenate([root, body, lhand, rhand, jaw, shape, expr])
    
    # Process to 133 dims (same as load_data.py)
    clip_poses = clip_poses[:, (3+3*11):]  # Remove first 36 → 143 dims
    clip_poses = np.concatenate([clip_poses[:, :-20], clip_poses[:, -10:]], axis=1)  # Remove shape → 133 dims
    
    return clip_poses.astype(np.float32)


def main():
    print("="*50)
    print("Generating Motion Tokens (using full model)")
    print("="*50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load mean/std and process to 133 dims
    mean_full = torch.load('./smpl-x/mean.pt')
    std_full = torch.load('./smpl-x/std.pt')
    
    mean_np = mean_full.numpy()[(3+3*11):]
    mean_np = np.concatenate([mean_np[:-20], mean_np[-10:]])
    std_np = std_full.numpy()[(3+3*11):]
    std_np = np.concatenate([std_np[:-20], std_np[-10:]])
    
    mean = torch.from_numpy(mean_np).float().to(device)
    std = torch.from_numpy(std_np).float().to(device)
    print(f"Processed mean/std: {mean.shape}")
    
    # Load full model to get the VAEs
    print("\nLoading tokenizer VAEs...")
    ckpt = torch.load('./tokenizer.ckpt', map_location='cpu', weights_only=False)
    
    # Get VAE configs from the checkpoint
    from mGPT.archs.mgpt_vq import VQVae
    
    # Body VAE: 43 features (re96 config)
    vae = VQVae(nfeats=43, code_num=96, code_dim=512, output_emb_width=512,
                down_t=2, stride_t=2, width=512, depth=3, dilation_growth_rate=3,
                norm=None, activation='relu', quantizer='ema_reset').to(device)
    
    # Hand VAE: 45 features (hand192 config)  
    hand_vae = VQVae(nfeats=45, code_num=192, code_dim=512, output_emb_width=512,
                     down_t=2, stride_t=2, width=512, depth=3, dilation_growth_rate=3,
                     norm=None, activation='relu', quantizer='ema_reset').to(device)
    
    rhand_vae = VQVae(nfeats=45, code_num=192, code_dim=512, output_emb_width=512,
                      down_t=2, stride_t=2, width=512, depth=3, dilation_growth_rate=3,
                      norm=None, activation='relu', quantizer='ema_reset').to(device)
    
    # Load weights
    state_dict = ckpt['state_dict']
    
    vae_dict = {k.replace('vae.', ''): v for k, v in state_dict.items() if k.startswith('vae.') and not k.startswith('vae.hand') and not k.startswith('vae.rhand')}
    hand_dict = {k.replace('hand_vae.', ''): v for k, v in state_dict.items() if k.startswith('hand_vae.')}
    rhand_dict = {k.replace('rhand_vae.', ''): v for k, v in state_dict.items() if k.startswith('rhand_vae.')}
    
    vae.load_state_dict(vae_dict, strict=False)
    hand_vae.load_state_dict(hand_dict, strict=False)
    rhand_vae.load_state_dict(rhand_dict, strict=False)
    
    vae.eval()
    hand_vae.eval()
    rhand_vae.eval()
    print("VAEs loaded!")
    
    # Output directory
    output_dir = Path('./data/TOKENS_h2s_csl_phoenix/how2sign')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split}...")
        
        csv_path = f'data/{split}/re_aligned/how2sign_realigned_{split}_preprocessed_fps.csv'
        if not os.path.exists(csv_path):
            print(f"  Skipping {split} - CSV not found")
            continue
            
        df = pd.read_csv(csv_path)
        pose_dir = Path(f'data/{split}/poses')
        
        success = 0
        failed = 0
        
        for idx in tqdm(range(len(df)), desc=split):
            name = df.iloc[idx]['SENTENCE_NAME']
            output_path = output_dir / f'{name}.npy'
            
            if output_path.exists():
                success += 1
                continue
            
            poses = load_pose_sample(pose_dir, name)
            if poses is None or len(poses) < 4:
                failed += 1
                continue
            
            try:
                # Normalize
                poses_tensor = torch.from_numpy(poses).float().to(device)
                poses_tensor = (poses_tensor - mean) / (std + 1e-8)
                poses_tensor = poses_tensor.unsqueeze(0)  # [1, T, 133]
                
                # Split features (same as model)
                # Body: [:30] + [120:] = 30 + 13 = 43
                # Lhand: [30:75] = 45
                # Rhand: [75:120] = 45
                feats_body = torch.cat([poses_tensor[..., :30], poses_tensor[..., 120:]], dim=-1)  # [1, T, 43]
                feats_lhand = poses_tensor[..., 30:75]  # [1, T, 45]
                feats_rhand = poses_tensor[..., 75:120]  # [1, T, 45]
                
                with torch.no_grad():
                    # Encode each part
                    code_body, _ = vae.encode(feats_body)  # [1, T/4]
                    code_lhand, _ = hand_vae.encode(feats_lhand)
                    code_rhand, _ = rhand_vae.encode(feats_rhand)
                    
                    # Stack: [T/4, 3] - body, lhand, rhand per timestep
                    tokens = torch.stack([code_body[0], code_lhand[0], code_rhand[0]], dim=-1)  # [T/4, 3]
                
                # Save with batch dim: [1, T/4, 3]
                tokens_np = tokens.cpu().numpy()[np.newaxis, ...]
                np.save(output_path, tokens_np)
                success += 1
                
            except Exception as e:
                failed += 1
                if failed <= 3:
                    print(f"\n  Error {name}: {e}")
        
        print(f"\n  {split}: {success} success, {failed} failed")
    
    print("\n" + "="*50)
    print("Done!")
    print("="*50)


if __name__ == '__main__':
    main()
