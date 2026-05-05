#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

model_dir = "models"
sys.path.append(model_dir)

from TimeDART_v2 import Model


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    use_gpu       = True
    use_multi_gpu = False
    gpu           = 0
    devices       = "0"

    @property
    def device(self):
        if self.use_gpu and torch.cuda.is_available():
            return torch.device(f"cuda:{self.gpu}")
        return torch.device("cpu")

    train_path  = os.path.join("DATA", "sample_QH", "self-supervused_data", "train", "train_samples_TS5.npy")
    test_path   = os.path.join("DATA", "sample_QH", "self-supervused_data", "val",   "val_samples_TS5.npy")
    window_size = 450
    step_size   = 450

    batch_size    = 128
    epochs        = 200
    learning_rate = 6e-5
    num_workers   = 6 if sys.platform != "win32" else 0
    log_interval  = 20
    save_dir      = "./checkpoints"
    patience      = 40

    task_name    = "pretrain"
    input_len    = 450
    pred_len     = 450
    enc_in       = 5
    c_out        = 5
    d_model      = 256
    n_heads      = 4
    d_ff         = 512
    dropout      = 0.2
    head_dropout = 0.2
    patch_len    = 16
    stride       = 16
    time_steps   = 2000
    scheduler    = "cosine"
    mask_ratio   = 0.3
    e_layers     = 3
    d_layers     = 3
    backbone     = "Transformer"
    llm_path     = None
    use_norm     = False

    noise_loss_scale = 1.0
    recon_loss_scale = 0.01
    adaptive_weights = True
    noise_weight     = 0.35
    recon_weight     = 0.35

    resume_training = False
    checkpoint_path = "./checkpoints/best_model.pth"

    confidence_threshold        = 0.94
    enable_confidence_filtering = True

    stage1_only          = False
    enable_visualization = True

    training_stage    = "denoising"
    stage1_epochs     = 300
    stage2_epochs     = 200
    stage1_checkpoint = "./checkpoints/stage1_classification.pth"

    enable_smoothing_constraints = True
    inner_smooth_weight          = 1.0
    boundary_align_weight        = 1.0
    anchor_extend_length         = 2
    smoothing_loss_weight        = 1.5


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DenoisingDataset(Dataset):
    def __init__(self, data_path: str, window_size: int = 450):
        self.signals = np.load(data_path)
        assert self.signals.ndim == 3, \
            f"Expected 3-D array [samples, time, channels], got {self.signals.ndim}-D."
        assert self.signals.shape[1] == window_size, \
            f"Window size mismatch: data={self.signals.shape[1]}, expected={window_size}."
        assert self.signals.shape[2] == 5, \
            f"Channel count mismatch: data={self.signals.shape[2]}, expected=5."

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.signals[idx].astype(np.float32))
        return x, x


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def load_checkpoint(model, optimizer, scheduler, checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}; training from scratch.")
        return 0, float('inf')
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    if 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'scheduler_state_dict' in ckpt and scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt.get('epoch', 0) + 1
    best_loss   = ckpt.get('best_test_loss', float('inf'))
    print(f"Checkpoint loaded (epoch {start_epoch - 1}) | best val loss: {best_loss:.6f}")
    return start_epoch, best_loss


def save_checkpoint(model, optimizer, scheduler, epoch: int, best_loss: float,
                    save_path: str, loss_components=None, is_best: bool = False) -> None:
    ckpt = {
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_test_loss':       best_loss,
    }
    if scheduler is not None:
        ckpt['scheduler_state_dict'] = scheduler.state_dict()
    if loss_components is not None:
        ckpt['best_loss_components'] = loss_components
    torch.save(ckpt, save_path)
    print(f"{'Best model' if is_best else 'Checkpoint'} saved to {save_path}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
    set_seed(42)
    args = Config()

    if args.use_gpu and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices if args.use_multi_gpu else str(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(DenoisingDataset(args.train_path, args.window_size), shuffle=True,  **loader_kw)
    test_loader  = DataLoader(DenoisingDataset(args.test_path,  args.window_size), shuffle=False, **loader_kw)

    model = Model(args).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    start_epoch    = 0
    best_test_loss = float('inf')

    if args.resume_training:
        start_epoch, best_test_loss = load_checkpoint(model, optimizer, scheduler, args.checkpoint_path)
        if start_epoch >= args.epochs:
            print(f"Training already complete (max epochs={args.epochs}, best val loss={best_test_loss:.6f}).")
            return

    train_losses, val_losses           = [], []
    train_noise_losses, val_noise_losses = [], []
    train_recon_losses, val_recon_losses = [], []

    loss_csv_path = os.path.join(args.save_dir, 'training_losses.csv')
    if args.resume_training and start_epoch > 0 and os.path.exists(loss_csv_path):
        try:
            df = pd.read_csv(loss_csv_path)
            df = df[df['epoch'] < start_epoch]
            if len(df) > 0:
                train_losses       = df['train_total_loss'].tolist()
                val_losses         = df['val_total_loss'].tolist()
                train_noise_losses = df['train_noise_loss'].tolist()
                train_recon_losses = df['train_recon_loss'].tolist()
                val_noise_losses   = df['val_noise_loss'].tolist()
                val_recon_losses   = df['val_recon_loss'].tolist()
        except Exception as e:
            print(f"Failed to load previous loss history: {e}")

    os.makedirs(args.save_dir, exist_ok=True)
    early_stop_counter = 0
    noise_loss_ema = recon_loss_ema = None
    ema_decay      = 0.9
    noise_weight   = args.noise_weight
    recon_weight   = args.recon_weight

    if args.training_stage == "classification":
        total_epochs = args.stage1_epochs
        model.training_stage = "classification"
        print(f"Stage 1 – classifier training ({total_epochs} epochs)")
    elif args.training_stage == "denoising":
        total_epochs = args.stage2_epochs
        model.training_stage = "denoising"
        print(f"Stage 2 – denoiser training ({total_epochs} epochs)")
        if os.path.exists(args.stage1_checkpoint):
            ckpt = torch.load(args.stage1_checkpoint, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"Stage-1 weights loaded from {args.stage1_checkpoint}")
        else:
            print(f"Warning: stage-1 checkpoint not found at {args.stage1_checkpoint}")
    else:
        total_epochs = args.epochs
        model.training_stage = "joint"
        print(f"Joint training ({total_epochs} epochs)")

    print(f"Best val loss so far: {best_test_loss} | noise_weight={noise_weight:.2f} recon_weight={recon_weight:.2f}")
    print("=" * 60)

    for epoch in range(start_epoch, total_epochs):
        model.train()
        train_total_loss = train_total_noise = train_total_recon = 0.0

        for batch_idx, (noisy, target) in enumerate(train_loader):
            noisy, target = noisy.to(device), target.to(device)

            out = model(noisy)
            if len(out) == 2:
                _, loss_tuple       = out
                total_loss          = loss_tuple[0]
                noise_loss          = loss_tuple[1]
                recon_loss          = loss_tuple[2]
                classification_loss = loss_tuple[3] if len(loss_tuple) > 3 else torch.tensor(0.0, device=device)
                n, r = noise_loss.item(), recon_loss.item()
                noise_loss_ema = n if noise_loss_ema is None else ema_decay * noise_loss_ema + (1 - ema_decay) * n
                recon_loss_ema = r if recon_loss_ema is None else ema_decay * recon_loss_ema + (1 - ema_decay) * r
            else:
                recon_loss          = nn.MSELoss()(out, target)
                noise_loss          = torch.tensor(0.0, device=device)
                classification_loss = torch.tensor(0.0, device=device)
                total_loss          = 0.5 * noise_loss + 0.5 * recon_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = noisy.size(0)
            train_total_loss  += total_loss.item() * bs
            train_total_noise += noise_loss.item() * bs
            train_total_recon += recon_loss.item() * bs

            if (batch_idx + 1) % args.log_interval == 0:
                weight_info = ""
                if hasattr(model, 'current_weights') and args.training_stage == "denoising":
                    w = model.current_weights
                    weight_info = (f" | W: N={w['noise_weight']:.1f} R={w['recon_weight']:.1f}"
                                   f" ratio={w['loss_ratio']:.1f}")
                cls_part = (f" | Cls Loss: {classification_loss.item():.4f}"
                            if args.enable_confidence_filtering and classification_loss.item() > 0 else "")
                print(
                    f"Epoch [{epoch+1}/{total_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] | "
                    f"Total: {total_loss.item():.4f} | "
                    f"Noise: {noise_loss.item():.4f} | "
                    f"Recon: {recon_loss.item():.4f}{cls_part}{weight_info}"
                )

        n_train = len(train_loader.dataset)
        train_avg_loss  = train_total_loss  / n_train
        train_avg_noise = train_total_noise / n_train
        train_avg_recon = train_total_recon / n_train
        train_losses.append(train_avg_loss)
        train_noise_losses.append(train_avg_noise)
        train_recon_losses.append(train_avg_recon)
        scheduler.step()

        model.eval()
        test_total_loss = test_total_noise = test_total_recon = 0.0
        with torch.no_grad():
            for noisy, target in test_loader:
                noisy, target = noisy.to(device), target.to(device)
                model.train()
                out = model(noisy)
                model.eval()
                if len(out) == 2:
                    _, loss_tuple = out
                    v_loss    = loss_tuple[0]
                    v_noise   = loss_tuple[1]
                    v_recon   = loss_tuple[2]
                else:
                    v_recon = nn.MSELoss()(out, target)
                    v_noise = torch.tensor(0.0, device=device)
                    v_loss  = v_recon
                bs = noisy.size(0)
                test_total_loss  += v_loss.item()  * bs
                test_total_noise += v_noise.item() * bs
                test_total_recon += v_recon.item() * bs

        n_val = len(test_loader.dataset)
        test_avg_loss  = test_total_loss  / n_val
        test_avg_noise = test_total_noise / n_val
        test_avg_recon = test_total_recon / n_val
        val_losses.append(test_avg_loss)
        val_noise_losses.append(test_avg_noise)
        val_recon_losses.append(test_avg_recon)

        print(
            f"\nEpoch [{epoch+1}/{total_epochs}] "
            f"Train total={train_avg_loss:.4f} noise={train_avg_noise:.4f} recon={train_avg_recon:.4f} | "
            f"Val total={test_avg_loss:.4f} noise={test_avg_noise:.4f} recon={test_avg_recon:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e}\n"
        )

        if test_avg_loss < best_test_loss:
            best_test_loss = test_avg_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_test_loss,
                os.path.join(args.save_dir, "best_model.pth"),
                {'noise_loss': test_avg_noise, 'recon_loss': test_avg_recon},
                is_best=True,
            )
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                print(f"Early stopping triggered after {args.patience} epochs without improvement.")
                break

        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_test_loss,
                os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth"),
            )

    if args.training_stage == "classification":
        save_checkpoint(model, optimizer, scheduler, total_epochs - 1, best_test_loss,
                        args.stage1_checkpoint)
        print(f"Stage 1 complete. Classifier saved to {args.stage1_checkpoint} | best loss: {best_test_loss:.4f}")
        print("Next: set training_stage = 'denoising' to run stage 2.")
    elif args.training_stage == "denoising":
        print(f"Stage 2 complete. Best denoising loss: {best_test_loss:.4f}")
    else:
        print(f"Joint training complete. Best total loss: {best_test_loss:.4f}")

    df = pd.DataFrame({
        'epoch':             range(1, len(train_losses) + 1),
        'train_total_loss':  train_losses,
        'val_total_loss':    val_losses,
        'train_noise_loss':  train_noise_losses,
        'val_noise_loss':    val_noise_losses,
        'train_recon_loss':  train_recon_losses,
        'val_recon_loss':    val_recon_losses,
    })
    df.to_csv(os.path.join(args.save_dir, 'training_losses.csv'), index=False)
    print(f"Loss history saved to {os.path.join(args.save_dir, 'training_losses.csv')}")


if __name__ == "__main__":
    train()