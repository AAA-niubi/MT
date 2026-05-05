import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NoiseRegionSmoothingConstraints(nn.Module):
    """
    噪声区域平滑约束模块
    基于锚点的平滑过渡约束，适配扩散模型
    """
    
    def __init__(self, 
                 inner_smooth_weight=1.0,      # 内部平滑约束权重
                 boundary_align_weight=1.0,    # 边界对齐约束权重
                 anchor_extend_length=2):      # 锚点扩展长度
        super(NoiseRegionSmoothingConstraints, self).__init__()
        
        self.inner_smooth_weight = inner_smooth_weight
        self.boundary_align_weight = boundary_align_weight
        self.anchor_extend_length = anchor_extend_length
        
    def find_noise_regions(self, noise_mask):
        """
        找到连续的噪声区域
        Args:
            noise_mask: [batch_size, time_steps, channels] bool tensor
        Returns:
            List of noise regions for each batch and channel
        """
        batch_size, time_steps, channels = noise_mask.shape
        all_regions = []
        
        for b in range(batch_size):
            batch_regions = []
            for c in range(channels):
                channel_mask = noise_mask[b, :, c].cpu().numpy()
                regions = []
                
                # 找连续的噪声区域
                in_noise = False
                start = 0
                
                for i, is_noise in enumerate(channel_mask):
                    if is_noise and not in_noise:
                        start = i
                        in_noise = True
                    elif not is_noise and in_noise:
                        regions.append((start, i-1))
                        in_noise = False
                
                if in_noise:
                    regions.append((start, time_steps-1))
                
                batch_regions.append(regions)
            all_regions.append(batch_regions)
        
        return all_regions
    
    def get_anchor_points(self, signal, noise_regions, batch_idx, channel_idx):
        """
        获取噪声区域的锚点
        Args:
            signal: [batch_size, time_steps, channels]
            noise_regions: 噪声区域列表 [(start, end), ...]
            batch_idx, channel_idx: 当前处理的批次和通道索引
        Returns:
            anchors: List of (pre_anchor_idx, post_anchor_idx, region_start, region_end)
        """
        time_steps = signal.shape[1]
        anchors = []
        
        for start, end in noise_regions:
            # 前锚点：噪声区域前的干净信号点
            pre_anchor_idx = max(0, start - 1)
            
            # 后锚点：噪声区域后的干净信号点  
            post_anchor_idx = min(time_steps - 1, end + 1)
            
            # 确保锚点不在噪声区域内
            if pre_anchor_idx >= start:
                pre_anchor_idx = max(0, start - self.anchor_extend_length)
            if post_anchor_idx <= end:
                post_anchor_idx = min(time_steps - 1, end + self.anchor_extend_length)
            
            anchors.append((pre_anchor_idx, post_anchor_idx, start, end))
        
        return anchors
    
    def compute_inner_smoothing_loss(self, signal, noise_regions, batch_idx, channel_idx):
        """
        计算噪声区域内部的平滑约束损失（增强版）
        L_inner = Σ|x_{i+1} - x_i|^2 + Σ|x_{i+2} - 2*x_{i+1} + x_i|^2 (二阶差分)
        """
        total_loss = 0.0
        count = 0
        
        for start, end in noise_regions:
            if end > start:  # 确保区域至少有2个点
                # 一阶差分约束（相邻点平滑）
                for i in range(start, end):
                    if i + 1 <= end:
                        diff = torch.abs(signal[batch_idx, i+1, channel_idx] - signal[batch_idx, i, channel_idx])
                        total_loss += diff ** 2  # 平方惩罚，增强约束
                        count += 1
                
                # 二阶差分约束（曲率平滑）- 增强版
                for i in range(start, end-1):
                    if i + 2 <= end:
                        second_diff = torch.abs(
                            signal[batch_idx, i+2, channel_idx] - 
                            2 * signal[batch_idx, i+1, channel_idx] + 
                            signal[batch_idx, i, channel_idx]
                        )
                        total_loss += second_diff ** 2 * 2.0  # 更强的二阶约束
                        count += 1
        
        return total_loss / max(count, 1)  # 避免除零
    
    def compute_boundary_alignment_loss(self, signal, anchors, batch_idx, channel_idx):
        """
        计算边界对齐约束损失（增强版）
        确保噪声区域边界与锚点平滑衔接，包含值对齐和斜率对齐
        """
        total_loss = 0.0
        count = 0
        
        for pre_anchor_idx, post_anchor_idx, start, end in anchors:
            # 前边界约束（增强版）
            if pre_anchor_idx >= 0 and start < signal.shape[1]:
                # 1. 值连续性约束
                value_diff = torch.abs(signal[batch_idx, start, channel_idx] - signal[batch_idx, pre_anchor_idx, channel_idx])
                total_loss += value_diff ** 2 * 3.0  # 强化值连续性
                count += 1
                
                # 2. 斜率连续性约束
                if pre_anchor_idx > 0:
                    k_pre = signal[batch_idx, pre_anchor_idx, channel_idx] - signal[batch_idx, pre_anchor_idx-1, channel_idx]
                    actual_slope = signal[batch_idx, start, channel_idx] - signal[batch_idx, pre_anchor_idx, channel_idx]
                    slope_diff = torch.abs(actual_slope - k_pre)
                    total_loss += slope_diff ** 2 * 2.0  # 强化斜率连续性
                    count += 1
                
                # 3. 多点平滑过渡约束
                if start + 1 < signal.shape[1]:
                    transition_slope = signal[batch_idx, start+1, channel_idx] - signal[batch_idx, start, channel_idx]
                    anchor_slope = signal[batch_idx, start, channel_idx] - signal[batch_idx, pre_anchor_idx, channel_idx]
                    transition_diff = torch.abs(transition_slope - anchor_slope)
                    total_loss += transition_diff ** 2 * 1.5  # 过渡平滑约束
                    count += 1
            
            # 后边界约束（增强版）
            if post_anchor_idx < signal.shape[1] and end >= 0:
                # 1. 值连续性约束
                value_diff = torch.abs(signal[batch_idx, end, channel_idx] - signal[batch_idx, post_anchor_idx, channel_idx])
                total_loss += value_diff ** 2 * 3.0  # 强化值连续性
                count += 1
                
                # 2. 斜率连续性约束
                if post_anchor_idx < signal.shape[1] - 1:
                    k_post = signal[batch_idx, post_anchor_idx+1, channel_idx] - signal[batch_idx, post_anchor_idx, channel_idx]
                    actual_slope = signal[batch_idx, post_anchor_idx, channel_idx] - signal[batch_idx, end, channel_idx]
                    slope_diff = torch.abs(actual_slope - k_post)
                    total_loss += slope_diff ** 2 * 2.0  # 强化斜率连续性
                    count += 1
                
                # 3. 多点平滑过渡约束
                if end - 1 >= 0:
                    transition_slope = signal[batch_idx, end, channel_idx] - signal[batch_idx, end-1, channel_idx]
                    anchor_slope = signal[batch_idx, post_anchor_idx, channel_idx] - signal[batch_idx, end, channel_idx]
                    transition_diff = torch.abs(transition_slope - anchor_slope)
                    total_loss += transition_diff ** 2 * 1.5  # 过渡平滑约束
                    count += 1
        
        return total_loss / max(count, 1)
    
    def forward(self, signal, noise_mask):
        """
        前向传播：计算平滑约束损失
        Args:
            signal: [batch_size, time_steps, channels] 去噪后的信号
            noise_mask: [batch_size, time_steps, channels] 噪声掩码
        Returns:
            smoothing_loss: 总的平滑约束损失
            loss_components: 损失组件字典
        """
        batch_size, time_steps, channels = signal.shape
        
        # 找到所有噪声区域
        all_noise_regions = self.find_noise_regions(noise_mask)
        
        total_inner_loss = 0.0
        total_boundary_loss = 0.0
        total_regions = 0
        
        # 对每个批次和通道计算约束损失
        for b in range(batch_size):
            for c in range(channels):
                noise_regions = all_noise_regions[b][c]
                
                if len(noise_regions) > 0:
                    # 获取锚点
                    anchors = self.get_anchor_points(signal, noise_regions, b, c)
                    
                    # 计算内部平滑损失
                    inner_loss = self.compute_inner_smoothing_loss(signal, noise_regions, b, c)
                    total_inner_loss += inner_loss
                    
                    # 计算边界对齐损失
                    boundary_loss = self.compute_boundary_alignment_loss(signal, anchors, b, c)
                    total_boundary_loss += boundary_loss
                    
                    total_regions += len(noise_regions)
        
        # 归一化损失
        if total_regions > 0:
            avg_inner_loss = total_inner_loss / total_regions
            avg_boundary_loss = total_boundary_loss / total_regions
        else:
            avg_inner_loss = torch.tensor(0.0, device=signal.device)
            avg_boundary_loss = torch.tensor(0.0, device=signal.device)
        
        # 加权组合
        total_smoothing_loss = (self.inner_smooth_weight * avg_inner_loss + 
                               self.boundary_align_weight * avg_boundary_loss)
        
        loss_components = {
            'inner_smoothing_loss': avg_inner_loss,
            'boundary_alignment_loss': avg_boundary_loss,
            'total_smoothing_loss': total_smoothing_loss,
            'processed_regions': total_regions
        }
        
        return total_smoothing_loss, loss_components
    
    def apply_smoothing_postprocess(self, signal, noise_mask, smoothing_strength=0.5):
        """
        应用平滑后处理（推理时使用）
        Args:
            signal: [batch_size, time_steps, channels]
            noise_mask: [batch_size, time_steps, channels]
            smoothing_strength: 平滑强度 (0-1)
        Returns:
            smoothed_signal: 平滑后的信号
        """
        smoothed_signal = signal.clone()
        batch_size, time_steps, channels = signal.shape
        
        # 找到所有噪声区域
        all_noise_regions = self.find_noise_regions(noise_mask)
        
        for b in range(batch_size):
            for c in range(channels):
                noise_regions = all_noise_regions[b][c]
                
                for start, end in noise_regions:
                    if end > start:
                        # 获取锚点值
                        pre_anchor_idx = max(0, start - 1)
                        post_anchor_idx = min(time_steps - 1, end + 1)
                        
                        pre_value = signal[b, pre_anchor_idx, c]
                        post_value = signal[b, post_anchor_idx, c]
                        
                        # 在噪声区域内进行线性插值平滑
                        region_length = end - start + 1
                        for i, pos in enumerate(range(start, end + 1)):
                            # 线性插值权重
                            alpha = i / max(region_length - 1, 1)
                            interpolated_value = pre_value * (1 - alpha) + post_value * alpha
                            
                            # 与原值混合
                            original_value = signal[b, pos, c]
                            smoothed_value = (original_value * (1 - smoothing_strength) + 
                                            interpolated_value * smoothing_strength)
                            
                            smoothed_signal[b, pos, c] = smoothed_value
        
        return smoothed_signal


def create_smoothing_constraints(inner_smooth_weight=1.0, 
                               boundary_align_weight=1.0,
                               anchor_extend_length=2):
    """
    创建平滑约束模块的工厂函数
    """
    return NoiseRegionSmoothingConstraints(
        inner_smooth_weight=inner_smooth_weight,
        boundary_align_weight=boundary_align_weight,
        anchor_extend_length=anchor_extend_length
    )
