import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from layers.TimeDART_EncDec import (
    ChannelIndependence,
    AddSosTokenAndDropLast,
    CausalTransformer,
    Diffusion,
    DenoisingPatchDecoder,
    ARFlattenHead,
)
from layers.Embed import Patch, PatchEmbedding, PositionalEncoding
from layers.SmoothingConstraints import NoiseRegionSmoothingConstraints


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_len = args.input_len
        self.task_name = args.task_name

        self.d_model = args.d_model
        self.num_heads = args.n_heads
        self.feedforward_dim = args.d_ff
        self.dropout = args.dropout

        self.patch_len = args.patch_len
        self.stride = args.stride
        self.seq_len = int((self.input_len - self.patch_len) / self.stride) + 1

        self.confidence_threshold = getattr(args, "confidence_threshold", 0.5)
        self.enable_confidence_filtering = getattr(args, "enable_confidence_filtering", True)
        self.training_stage = getattr(args, "training_stage", "joint")

        self.enable_smoothing_constraints = getattr(args, "enable_smoothing_constraints", True)
        self.inner_smooth_weight = getattr(args, "inner_smooth_weight", 1.0)
        self.boundary_align_weight = getattr(args, "boundary_align_weight", 1.0)
        self.anchor_extend_length = getattr(args, "anchor_extend_length", 2)
        self.smoothing_loss_weight = getattr(args, "smoothing_loss_weight", 0.1)

        self.channel_independence = ChannelIndependence()
        self.patch = Patch(patch_len=self.patch_len, stride=self.stride)
        self.enc_embedding = PatchEmbedding(patch_len=self.patch_len, d_model=self.d_model)
        self.positional_encoding = PositionalEncoding(d_model=self.d_model, dropout=self.dropout)

        self.device = self._acquire_device()

        self.sos_token = nn.Parameter(
            torch.randn(1, 1, self.d_model, device=self.device), requires_grad=True
        )
        self.add_sos_token_and_drop_last = AddSosTokenAndDropLast(sos_token=self.sos_token)

        self.encoder = self._init_encoder()

        self.diffusion = Diffusion(
            time_steps=args.time_steps,
            device=self.device,
            scheduler=args.scheduler,
        )

        self.denoising_patch_decoder = DenoisingPatchDecoder(
            d_model=args.d_model,
            num_layers=args.d_layers,
            num_heads=args.n_heads,
            feedforward_dim=args.d_ff,
            dropout=args.dropout,
            mask_ratio=args.mask_ratio,
        )

        self.projection = ARFlattenHead(
            d_model=self.d_model,
            patch_len=self.patch_len,
            dropout=args.head_dropout,
        )
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.0)

        self.smoothing_constraints = (
            NoiseRegionSmoothingConstraints(
                inner_smooth_weight=self.inner_smooth_weight,
                boundary_align_weight=self.boundary_align_weight,
                anchor_extend_length=self.anchor_extend_length,
            ).to(self.device)
            if self.enable_smoothing_constraints
            else None
        )

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            return torch.device(f"cuda:{self.args.gpu}")
        return torch.device("cpu")

    def _init_encoder(self):
        if self.args.backbone == "Qwen2.5-0.5B":
            return AutoModelForCausalLM.from_pretrained(
                self.args.llm_path,
                output_attentions=True,
                output_hidden_states=True,
                device_map=self.device,
                trust_remote_code=True,
                attn_implementation="eager",
            )
        elif self.args.backbone == "Transformer":
            return CausalTransformer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                num_layers=self.args.e_layers,
                feedforward_dim=self.feedforward_dim,
                dropout=self.dropout,
            )
        raise ValueError(f"Unsupported backbone: {self.args.backbone}")

    def freeze_parameters(self, stage):
        if stage == "classification":
            for param in self.diffusion.parameters():
                param.requires_grad = False
            for param in self.denoising_patch_decoder.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.projection.parameters():
                param.requires_grad = True
            for param in self.enc_embedding.parameters():
                param.requires_grad = True

        elif stage == "denoising":
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.projection.parameters():
                param.requires_grad = False
            for param in self.enc_embedding.parameters():
                param.requires_grad = False
            for param in self.diffusion.parameters():
                param.requires_grad = True
            for param in self.denoising_patch_decoder.parameters():
                param.requires_grad = True

        elif stage == "joint":
            for param in self.parameters():
                param.requires_grad = True

    def _encode(self, x_embed):
        if self.args.backbone == "Qwen2.5-0.5B":
            return self.encoder(inputs_embeds=x_embed, output_hidden_states=True).hidden_states[-1]
        return self.encoder(x_embed, is_mask=True)

    def _embed(self, x_patch):
        emb = self.enc_embedding(x_patch)
        emb = self.add_sos_token_and_drop_last(emb)
        return self.positional_encoding(emb)

    def _align_to_input_len(self, tensor, input_len, batch_size, num_features):
        if tensor.dim() == 4:
            tensor = tensor.reshape(tensor.size(0), tensor.size(1), -1)
        if tensor.size(-1) != input_len or (tensor.dim() == 3 and tensor.size(1) != input_len):
            tensor = F.interpolate(
                tensor.transpose(1, 2), size=input_len, mode="linear", align_corners=False
            ).transpose(1, 2)
        return tensor.view(batch_size, input_len, num_features)

    def _classify(self, original_x, batch_size, num_features, input_len, no_grad=False):
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            x_patch = self.patch(self.channel_independence(original_x))
            features = self._encode(self._embed(x_patch))
            out = self.projection(features.reshape(batch_size, num_features, -1, self.d_model))
            out = self._align_to_input_len(out, input_len, batch_size, num_features)
        return torch.sigmoid(torch.abs(out - original_x))

    def _denoise(self, noise_regions_only, batch_size, num_features, input_len):
        noise_x_patch = self.patch(self.channel_independence(noise_regions_only))
        noise_x_patch_diffused, real_noise, _ = self.diffusion(noise_x_patch)

        noise_encoded = self._encode(self._embed(noise_x_patch))
        noise_diffused_emb = self.positional_encoding(self.enc_embedding(noise_x_patch_diffused))

        predict_patch = self.denoising_patch_decoder(
            query=noise_diffused_emb,
            key=noise_encoded,
            value=noise_encoded,
            is_tgt_mask=True,
            is_src_mask=True,
        )

        out = self.projection(predict_patch.reshape(batch_size, num_features, -1, self.d_model))
        out = self._align_to_input_len(out, input_len, batch_size, num_features)
        return out, noise_x_patch_diffused, real_noise

    def _apply_smoothing(self, denoised, noise_mask, training):
        if not (self.enable_smoothing_constraints and self.smoothing_constraints is not None):
            return denoised * noise_mask.float(), torch.tensor(0.0, device=denoised.device)
        if training:
            smoothing_loss, _ = self.smoothing_constraints(denoised, noise_mask)
            return denoised * noise_mask.float(), smoothing_loss
        smoothed = self.smoothing_constraints.apply_smoothing_postprocess(
            denoised, noise_mask, smoothing_strength=0.3
        )
        return smoothed * noise_mask.float(), torch.tensor(0.0, device=denoised.device)

    def _build_noise_labels(self, original_x, input_len):
        grad = torch.abs(original_x[:, 1:, :] - original_x[:, :-1, :])
        grad = F.pad(grad, (0, 0, 1, 0), mode="replicate")
        grad_labels = (grad > grad.mean(dim=1, keepdim=True) + 1.5 * (grad.std(dim=1, keepdim=True) + 1e-8)).float()

        window_size = 5
        padded = F.pad(original_x, (0, 0, window_size // 2, window_size // 2), mode="reflect")
        local_vars = torch.stack(
            [torch.var(padded[:, i:i + window_size, :], dim=1) for i in range(input_len)], dim=1
        )
        var_labels = (
            local_vars > local_vars.mean(dim=1, keepdim=True) + 1.0 * (local_vars.std(dim=1, keepdim=True) + 1e-8)
        ).float()

        fft_mag = torch.abs(torch.fft.rfft(original_x, dim=1))
        freq_len = fft_mag.size(1)
        high_freq_ratio = fft_mag[:, freq_len // 3:, :].sum(dim=1, keepdim=True) / (
            fft_mag.sum(dim=1, keepdim=True) + 1e-8
        )
        freq_labels = (high_freq_ratio.expand(-1, input_len, -1) > 0.3).float()

        return ((grad_labels + var_labels + freq_labels) / 3.0).detach()

    def _recon_loss(self, final_output, original_x, noise_mask):
        if noise_mask.sum() > 0:
            return (noise_mask.float() * (final_output - original_x) ** 2).sum() / noise_mask.sum()
        return nn.MSELoss()(final_output, original_x)

    def _noise_mask(self, noise_probability):
        if self.enable_confidence_filtering:
            return noise_probability > self.confidence_threshold
        return torch.ones_like(noise_probability, dtype=torch.bool)

    def pretrain(self, x):
        batch_size, input_len, num_features = x.size()
        original_x = x.clone()

        if self.training:
            self.freeze_parameters(self.training_stage)

        if self.training_stage == "classification":
            noise_probability = self._classify(original_x, batch_size, num_features, input_len)
            if self.training:
                noise_labels = self._build_noise_labels(original_x, input_len)
                cls_loss = nn.MSELoss()(noise_probability, noise_labels)
                p = noise_probability.mean()
                diversity_loss = -torch.log(p * (1 - p) + 1e-8)
                total_loss = cls_loss + 0.1 * diversity_loss
                return noise_probability, (total_loss, torch.tensor(0.0), torch.tensor(0.0), cls_loss)
            return noise_probability

        elif self.training_stage == "denoising":
            noise_probability = self._classify(original_x, batch_size, num_features, input_len, no_grad=True)
            noise_mask = self._noise_mask(noise_probability)

            denoised, noise_x_patch_diffused, real_noise = self._denoise(
                original_x * noise_mask.float(), batch_size, num_features, input_len
            )
            processed, _ = self._apply_smoothing(denoised, noise_mask, self.training)
            final_output = original_x * (~noise_mask).float() + processed

            if self.training:
                predicted_noise = noise_x_patch_diffused - self.patch(self.channel_independence(processed))
                noise_loss = nn.MSELoss()(predicted_noise, real_noise)
                recon_loss = self._recon_loss(final_output, original_x, noise_mask)
                noise_weight = getattr(self.args, "noise_weight", 0.5)
                recon_weight = getattr(self.args, "recon_weight", 0.5)
                total_loss = noise_weight * noise_loss + recon_weight * recon_loss
                return final_output, (total_loss, noise_loss, recon_loss, torch.tensor(0.0))
            return final_output

        else:
            noise_probability = self._classify(original_x, batch_size, num_features, input_len)
            noise_mask = self._noise_mask(noise_probability)

            denoised, noise_x_patch_diffused, real_noise = self._denoise(
                original_x * noise_mask.float(), batch_size, num_features, input_len
            )
            processed, smoothing_loss = self._apply_smoothing(denoised, noise_mask, self.training)
            final_output = original_x * (~noise_mask).float() + processed

            if self.training:
                recon_error = torch.abs(final_output - original_x)
                noise_labels = (recon_error > recon_error.mean(dim=1, keepdim=True)).float()
                cls_loss = nn.BCELoss()(noise_probability, noise_labels.detach())
                predicted_noise = noise_x_patch_diffused - self.patch(self.channel_independence(processed))
                noise_loss = nn.MSELoss()(predicted_noise, real_noise)
                recon_loss = self._recon_loss(final_output, original_x, noise_mask)
                total_loss = (
                    0.2 * cls_loss
                    + 0.3 * noise_loss
                    + 0.5 * recon_loss
                    + self.smoothing_loss_weight * smoothing_loss
                )
                return final_output, (total_loss, noise_loss, recon_loss, cls_loss)
            return final_output

    def forward(self, x, y=None):
        if self.task_name != "pretrain":
            raise ValueError(f"This model only supports 'pretrain' task, got: {self.task_name}")
        return self.pretrain(x)