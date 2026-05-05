
---

## Environment

| Dependency | Version |
|---|---|
| Python | 3.10.18 |
| PyTorch | 2.1.0 |
| CUDA | 11.8 |
| cuDNN | 8.7 |
| transformers | 4.38.0 |
| numpy | 1.26.4 |
| scipy | 1.12.0 |
| matplotlib | 3.8.3 |
| scikit-learn | 1.4.1 |

> Other CUDA versions (12.x) are also compatible. Adjust the PyTorch installation command accordingly.

--


## Data

Place your MT time series data under the `data/` directory. The expected format is `.npy` 

---

## Training

The pipeline supports three training stages: `classification`, `denoising`, and `joint`.

**Stage 1 — Train the noise region classifier:**

```bash
python train.py \
  --data_path data/train \
  --training_stage classification \
  --input_len 512 \
  --patch_len 16 \
  --stride 8 \
  --d_model 256 \
  --n_heads 8 \
  --d_ff 512 \
  --e_layers 4 \
  --d_layers 2 \
  --time_steps 1000 \
  --scheduler linear \
  --batch_size 32 \
  --epochs 50 \
  --lr 1e-4 \
  --backbone Transformer \
  --checkpoints_path results/checkpoints/stage1 \
  --use_gpu True \
  --gpu 0
```

**Stage 2 — Train the denoiser:**

```bash
python train.py \
  --data_path data/train \
  --training_stage denoising \
  --pretrained_path results/checkpoints/stage1/best_model.pth \
  --checkpoints_path results/checkpoints/stage2 \
  --noise_weight 0.5 \
  --recon_weight 0.5 \
  --epochs 50 \
  --lr 5e-5 \
  [... same architecture args as Stage 1 ...]
```

## Intermediate Outputs

During training, visualization results are automatically saved.  You can inspect them directly without any additional steps 


## License

This project is released under the [MIT License](LICENSE).
