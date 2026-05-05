import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger('PhysicalConstraints')


class WaveformClassifier(nn.Module):
    """方波和三角波噪声分类器"""
    
    def __init__(self, d_model: int = 256, num_classes: int = 3):
        super(WaveformClassifier, self).__init__()
        self.d_model = d_model
        self.num_classes = num_classes  # 0: 正常信号, 1: 方波噪声, 2: 三角波噪声
        
        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, num_features] 或 [batch_size, seq_len]
        Returns:
            classification_logits: [batch_size, num_classes] 或 [batch_size, num_features, num_classes]
        """
        if x.dim() == 3:
            batch_size, seq_len, num_features = x.shape
            # 对每个通道分别分类
            x_reshaped = x.transpose(1, 2).contiguous()  # [batch, features, seq_len]
            results = []
            for i in range(num_features):
                channel_data = x_reshaped[:, i:i+1, :]  # [batch, 1, seq_len]
                features = self.feature_extractor(channel_data)  # [batch, 128, 1]
                features = features.squeeze(-1)  # [batch, 128]
                logits = self.classifier(features)  # [batch, num_classes]
                results.append(logits)
            return torch.stack(results, dim=1)  # [batch, num_features, num_classes]
        else:
            # 单通道处理
            x = x.unsqueeze(1)  # [batch, 1, seq_len]
            features = self.feature_extractor(x)
            features = features.squeeze(-1)
            return self.classifier(features)


class SquareWaveConstraint(nn.Module):
    """方波噪声物理约束"""
    
    def __init__(self, constraint_strength: float = 0.1):
        super(SquareWaveConstraint, self).__init__()
        self.constraint_strength = constraint_strength
        
    def detect_square_wave_features(self, x: torch.Tensor) -> torch.Tensor:
        """检测方波特征"""
        # 计算一阶导数
        grad = torch.diff(x, dim=1)
        grad_padded = F.pad(grad, (0, 0, 1, 0), mode='replicate')
        
        # 方波特征：导数的绝对值很大（跳跃）且变化稀疏
        grad_abs = torch.abs(grad_padded)
        grad_threshold = torch.quantile(grad_abs, 0.9, dim=1, keepdim=True)
        
        # 检测跳跃点
        jump_points = grad_abs > grad_threshold
        
        # 方波特征：跳跃点之间的区域应该相对平坦
        return jump_points.float()
    
    def apply_constraint(self, x: torch.Tensor, noise_mask: torch.Tensor) -> torch.Tensor:
        """应用方波约束"""
        square_features = self.detect_square_wave_features(x)
        
        # 在检测到方波特征的区域应用平滑约束
        constraint_mask = square_features * noise_mask
        
        if constraint_mask.sum() > 0:
            # 对方波区域进行分段常数约束
            smoothed = self._apply_piecewise_constant_constraint(x, constraint_mask)
            x = x * (1 - constraint_mask * self.constraint_strength) + \
                smoothed * (constraint_mask * self.constraint_strength)
        
        return x
    
    def _apply_piecewise_constant_constraint(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """应用分段常数约束"""
        batch_size, seq_len, num_features = x.shape
        result = x.clone()
        
        for b in range(batch_size):
            for f in range(num_features):
                channel_mask = mask[b, :, f]
                if channel_mask.sum() > 0:
                    # 找到连续的方波区域
                    regions = self._find_continuous_regions(channel_mask)
                    for start, end in regions:
                        if end - start > 2:  # 只处理足够长的区域
                            # 用区域的中位数替换
                            region_values = x[b, start:end+1, f]
                            median_val = torch.median(region_values)
                            result[b, start:end+1, f] = median_val
        
        return result
    
    def _find_continuous_regions(self, mask: torch.Tensor) -> list:
        """找到连续的True区域"""
        regions = []
        in_region = False
        start = 0
        
        for i, val in enumerate(mask):
            if val and not in_region:
                start = i
                in_region = True
            elif not val and in_region:
                regions.append((start, i-1))
                in_region = False
        
        if in_region:
            regions.append((start, len(mask)-1))
        
        return regions


class TriangleWaveConstraint(nn.Module):
    """三角波噪声物理约束"""
    
    def __init__(self, constraint_strength: float = 0.1):
        super(TriangleWaveConstraint, self).__init__()
        self.constraint_strength = constraint_strength
        
    def detect_triangle_wave_features(self, x: torch.Tensor) -> torch.Tensor:
        """检测三角波特征"""
        # 计算一阶和二阶导数
        grad1 = torch.diff(x, dim=1)
        grad1_padded = F.pad(grad1, (0, 0, 1, 0), mode='replicate')
        
        grad2 = torch.diff(grad1_padded, dim=1)
        grad2_padded = F.pad(grad2, (0, 0, 1, 0), mode='replicate')
        
        # 三角波特征：一阶导数相对恒定，二阶导数在转折点有跳跃
        grad1_var = self._local_variance(grad1_padded, window_size=5)
        grad2_abs = torch.abs(grad2_padded)
        
        # 低一阶导数方差 + 高二阶导数绝对值 = 三角波特征
        triangle_score = (1.0 / (grad1_var + 1e-6)) * grad2_abs
        triangle_threshold = torch.quantile(triangle_score, 0.8, dim=1, keepdim=True)
        
        return (triangle_score > triangle_threshold).float()
    
    def _local_variance(self, x: torch.Tensor, window_size: int = 5) -> torch.Tensor:
        """计算局部方差"""
        batch_size, seq_len, num_features = x.shape
        padded_x = F.pad(x, (0, 0, window_size//2, window_size//2), mode='reflect')
        
        variances = []
        for i in range(seq_len):
            window = padded_x[:, i:i+window_size, :]
            var = torch.var(window, dim=1)
            variances.append(var)
        
        return torch.stack(variances, dim=1)
    
    def apply_constraint(self, x: torch.Tensor, noise_mask: torch.Tensor) -> torch.Tensor:
        """应用三角波约束"""
        triangle_features = self.detect_triangle_wave_features(x)
        
        # 在检测到三角波特征的区域应用线性约束
        constraint_mask = triangle_features * noise_mask
        
        if constraint_mask.sum() > 0:
            # 对三角波区域进行分段线性约束
            linearized = self._apply_piecewise_linear_constraint(x, constraint_mask)
            x = x * (1 - constraint_mask * self.constraint_strength) + \
                linearized * (constraint_mask * self.constraint_strength)
        
        return x
    
    def _apply_piecewise_linear_constraint(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """应用分段线性约束"""
        batch_size, seq_len, num_features = x.shape
        result = x.clone()
        
        for b in range(batch_size):
            for f in range(num_features):
                channel_mask = mask[b, :, f]
                if channel_mask.sum() > 0:
                    # 找到连续的三角波区域
                    regions = self._find_continuous_regions(channel_mask)
                    for start, end in regions:
                        if end - start > 3:  # 只处理足够长的区域
                            # 用线性插值替换
                            start_val = x[b, start, f]
                            end_val = x[b, end, f]
                            length = end - start + 1
                            linear_vals = torch.linspace(start_val, end_val, length, device=x.device)
                            result[b, start:end+1, f] = linear_vals
        
        return result
    
    def _find_continuous_regions(self, mask: torch.Tensor) -> list:
        """找到连续的True区域"""
        regions = []
        in_region = False
        start = 0
        
        for i, val in enumerate(mask):
            if val and not in_region:
                start = i
                in_region = True
            elif not val and in_region:
                regions.append((start, i-1))
                in_region = False
        
        if in_region:
            regions.append((start, len(mask)-1))
        
        return regions


class PhysicalConstraintsModule(nn.Module):
    """物理约束主模块 - 只对已分类的噪声区域进行波形检测"""
    
    def __init__(self, 
                 d_model: int = 256,
                 enable_waveform_classification: bool = True,
                 square_wave_strength: float = 0.2,
                 triangle_wave_strength: float = 0.2,
                 classification_threshold: float = 0.7):
        super(PhysicalConstraintsModule, self).__init__()
        
        self.enable_waveform_classification = enable_waveform_classification
        self.classification_threshold = classification_threshold
        
        # 约束模块
        self.square_wave_constraint = SquareWaveConstraint(constraint_strength=square_wave_strength)
        self.triangle_wave_constraint = TriangleWaveConstraint(constraint_strength=triangle_wave_strength)
        
        # 约束权重（可学习）
        self.constraint_weights = nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)
        
    def classify_waveforms_in_noise_regions(self, x: torch.Tensor, noise_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """只在噪声区域内进行波形分类"""
        if not self.enable_waveform_classification:
            # 如果不启用分类，返回全零掩码
            batch_size, seq_len, num_features = x.shape
            device = x.device
            return (torch.zeros(batch_size, seq_len, num_features, device=device),
                    torch.zeros(batch_size, seq_len, num_features, device=device))
        
        batch_size, seq_len, num_features = x.shape
        square_wave_mask = torch.zeros(batch_size, seq_len, num_features, device=x.device)
        triangle_wave_mask = torch.zeros(batch_size, seq_len, num_features, device=x.device)
        
        # 只对噪声区域进行波形检测
        for b in range(batch_size):
            for f in range(num_features):
                # 获取当前通道的噪声掩码
                channel_noise_mask = noise_mask[b, :, f]
                
                if channel_noise_mask.sum() > 0:  # 如果有噪声区域
                    # 提取噪声区域的信号
                    channel_signal = x[b, :, f]
                    noise_signal = channel_signal[channel_noise_mask]
                    
                    if len(noise_signal) > 10:  # 确保有足够的数据点进行分析
                        # 检测方波特征
                        square_features = self.square_wave_constraint.detect_square_wave_features(
                            noise_signal.unsqueeze(0).unsqueeze(-1)
                        ).squeeze()
                        
                        # 检测三角波特征
                        triangle_features = self.triangle_wave_constraint.detect_triangle_wave_features(
                            noise_signal.unsqueeze(0).unsqueeze(-1)
                        ).squeeze()
                        
                        # 计算特征强度
                        square_strength = square_features.mean()
                        triangle_strength = triangle_features.mean()
                        
                        # 根据特征强度决定波形类型
                        if square_strength > self.classification_threshold:
                            square_wave_mask[b, channel_noise_mask, f] = 1.0
                        
                        if triangle_strength > self.classification_threshold:
                            triangle_wave_mask[b, channel_noise_mask, f] = 1.0
        
        return square_wave_mask, triangle_wave_mask
    
    def apply_constraints(self, 
                         x: torch.Tensor, 
                         noise_mask: torch.Tensor,
                         square_wave_mask: Optional[torch.Tensor] = None,
                         triangle_wave_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """应用物理约束"""
        
        # 如果没有提供波形掩码，则在噪声区域内进行分类
        if square_wave_mask is None or triangle_wave_mask is None:
            square_wave_mask, triangle_wave_mask = self.classify_waveforms_in_noise_regions(x, noise_mask)
        
        constrained_x = x.clone()
        
        # 应用方波约束（只在检测到方波的噪声区域）
        square_constraint_mask = square_wave_mask * noise_mask
        if square_constraint_mask.sum() > 0:
            constrained_x = self.square_wave_constraint.apply_constraint(
                constrained_x, square_constraint_mask
            )
        
        # 应用三角波约束（只在检测到三角波的噪声区域）
        triangle_constraint_mask = triangle_wave_mask * noise_mask
        if triangle_constraint_mask.sum() > 0:
            constrained_x = self.triangle_wave_constraint.apply_constraint(
                constrained_x, triangle_constraint_mask
            )
        
        # 使用可学习权重混合原始信号和约束后的信号
        weights = F.softmax(self.constraint_weights, dim=0)
        final_x = weights[0] * x + weights[1] * constrained_x
        
        return final_x
    
    def forward(self, 
                x: torch.Tensor, 
                noise_mask: torch.Tensor,
                return_classification: bool = False) -> torch.Tensor:
        """前向传播 - 只对已分类的噪声区域进行波形检测和约束"""
        
        # 在噪声区域内进行波形分类
        square_wave_mask, triangle_wave_mask = self.classify_waveforms_in_noise_regions(x, noise_mask)
        
        # 应用约束
        constrained_x = self.apply_constraints(x, noise_mask, square_wave_mask, triangle_wave_mask)
        
        if return_classification:
            # 返回分类结果（用于调试和可视化）
            return constrained_x, None, square_wave_mask, triangle_wave_mask
        else:
            return constrained_x
    
    def get_constraint_loss(self, 
                           x: torch.Tensor, 
                           constrained_x: torch.Tensor,
                           noise_mask: torch.Tensor) -> torch.Tensor:
        """计算约束损失"""
        
        # 只在噪声区域计算约束损失
        if noise_mask.sum() > 0:
            constraint_diff = (constrained_x - x) * noise_mask
            constraint_loss = torch.mean(constraint_diff ** 2)
        else:
            constraint_loss = torch.tensor(0.0, device=x.device)
        
        return constraint_loss


def create_synthetic_square_wave(length: int, amplitude: float = 1.0, period: int = 20) -> torch.Tensor:
    """创建合成方波用于测试"""
    t = torch.arange(length, dtype=torch.float32)
    square_wave = amplitude * torch.sign(torch.sin(2 * np.pi * t / period))
    return square_wave


def create_synthetic_triangle_wave(length: int, amplitude: float = 1.0, period: int = 20) -> torch.Tensor:
    """创建合成三角波用于测试"""
    t = torch.arange(length, dtype=torch.float32)
    triangle_wave = amplitude * (2 / np.pi) * torch.arcsin(torch.sin(2 * np.pi * t / period))
    return triangle_wave


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    batch_size, seq_len, num_features = 4, 100, 5
    
    # 正常信号
    normal_signal = torch.randn(batch_size, seq_len, num_features, device=device)
    
    # 添加方波和三角波噪声
    test_signal = normal_signal.clone()
    
    # 在某些区域添加方波噪声
    square_wave = create_synthetic_square_wave(seq_len).to(device)
    test_signal[0, :, 0] = square_wave
    test_signal[1, 20:60, 1] = square_wave[20:60]
    
    # 在某些区域添加三角波噪声
    triangle_wave = create_synthetic_triangle_wave(seq_len).to(device)
    test_signal[2, :, 2] = triangle_wave
    test_signal[3, 30:80, 3] = triangle_wave[30:80]
    
    # 创建噪声掩码
    noise_mask = torch.zeros(batch_size, seq_len, num_features, device=device)
    noise_mask[0, :, 0] = 1.0  # 方波区域
    noise_mask[1, 20:60, 1] = 1.0  # 方波区域
    noise_mask[2, :, 2] = 1.0  # 三角波区域
    noise_mask[3, 30:80, 3] = 1.0  # 三角波区域
    
    # 创建物理约束模块
    constraints_module = PhysicalConstraintsModule(
        d_model=256,
        enable_waveform_classification=True,
        square_wave_strength=0.3,
        triangle_wave_strength=0.3,
        classification_threshold=0.5
    ).to(device)
    
    # 测试前向传播
    with torch.no_grad():
        constrained_signal, classification_logits, square_mask, triangle_mask = constraints_module(
            test_signal, noise_mask, return_classification=True
        )
    
    print("测试完成！")
    print(f"输入信号形状: {test_signal.shape}")
    print(f"约束后信号形状: {constrained_signal.shape}")
    print(f"分类结果形状: {classification_logits.shape}")
    print(f"方波掩码检测到的区域数: {square_mask.sum().item()}")
    print(f"三角波掩码检测到的区域数: {triangle_mask.sum().item()}")
    
    # 计算约束效果
    constraint_effect = torch.mean(torch.abs(constrained_signal - test_signal) * noise_mask)
    print(f"约束效果 (平均变化): {constraint_effect.item():.6f}")
