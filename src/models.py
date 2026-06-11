# 模型模块 - TIPNet和MFFNet

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    通道注意力机制
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    空间注意力机制
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class TCBBlock(nn.Module):
    """
    舌特征基础注意力残差块
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, attention_ratio=1/3):
        super(TCBBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, 
                               padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 
                               padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 注意力机制
        self.channel_att = ChannelAttention(out_channels)
        self.spatial_att = SpatialAttention()
        self.attention_ratio = attention_ratio
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 应用注意力机制
        channel_att = self.channel_att(out)
        spatial_att = self.spatial_att(out)
        
        # 混合注意力
        att = self.attention_ratio * channel_att + (1 - self.attention_ratio) * spatial_att
        out = out * att
        
        # 残差连接
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out


class TCNBlock(nn.Module):
    """
    舌特征瓶颈注意力残差块
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 expansion=4, attention_ratio=1/3):
        super(TCNBlock, self).__init__()
        
        hidden_channels = out_channels // expansion
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 
                               padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 注意力机制
        self.channel_att = ChannelAttention(out_channels)
        self.spatial_att = SpatialAttention()
        self.attention_ratio = attention_ratio
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # 应用注意力机制
        channel_att = self.channel_att(out)
        spatial_att = self.spatial_att(out)
        
        # 混合注意力
        att = self.attention_ratio * channel_att + (1 - self.attention_ratio) * spatial_att
        out = out * att
        
        # 残差连接
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out


class TIPNet(nn.Module):
    """
    舌图像感知网络（Tongue Image Perception Network）
    多支路多尺度特征提取
    """
    def __init__(self, attention_ratio=1/3):
        super(TIPNet, self).__init__()
        
        # Stem卷积
        self.stem = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Branch 1: 3×3卷积（细粒度特征）
        self.branch1 = nn.Sequential(
            TCBBlock(192, 512, kernel_size=3, attention_ratio=attention_ratio),
            TCNBlock(512, 512, kernel_size=3, attention_ratio=attention_ratio),
            TCNBlock(512, 512, kernel_size=3, attention_ratio=attention_ratio)
        )
        
        # Branch 2: 5×5卷积（中尺度特征）
        self.branch2 = nn.Sequential(
            TCBBlock(192, 512, kernel_size=5, attention_ratio=attention_ratio),
            TCNBlock(512, 512, kernel_size=5, attention_ratio=attention_ratio),
            TCNBlock(512, 512, kernel_size=5, attention_ratio=attention_ratio)
        )
        
        # Branch 3: 7×7卷积（大尺度特征）
        self.branch3 = nn.Sequential(
            TCBBlock(192, 512, kernel_size=7, attention_ratio=attention_ratio),
            TCNBlock(512, 512, kernel_size=7, attention_ratio=attention_ratio),
            TCNBlock(512, 512, kernel_size=7, attention_ratio=attention_ratio)
        )
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入图像 (B, 3, 448, 448)
            
        Returns:
            torch.Tensor: 图像特征向量 (B, 1536)
        """
        # Stem
        x = self.stem(x)  # (B, 192, 112, 112)
        
        # 三个分支处理
        b1 = self.branch1(x)  # (B, 512, 56, 56)
        b2 = self.branch2(x)  # (B, 512, 56, 56)
        b3 = self.branch3(x)  # (B, 512, 56, 56)
        
        # 拼接分支输出
        features = torch.cat([b1, b2, b3], dim=1)  # (B, 1536, 56, 56)
        
        # 全局平均池化
        features = self.global_avg_pool(features)  # (B, 1536, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 1536)
        
        return features


class MFFNet(nn.Module):
    """
    多源特征融合网络（Multi-source Feature Fusion Network）
    融合图像特征和生理指标
    """
    def __init__(self, imagery_dim=1536, phys_dim=8, num_classes=2):
        super(MFFNet, self).__init__()
        
        self.imagery_dim = imagery_dim
        self.phys_dim = phys_dim
        
        # 性别和年龄的MLP（用于调整图像特征的通道权重）
        self.mlp_sex_age = nn.Sequential(
            nn.Linear(3, 128),  # 性别(1维) + 年龄(1维) + 偏置
            nn.ReLU(inplace=True),
            nn.Linear(128, imagery_dim)
        )
        
        # 其他生理指标的MLP（用于直接融合）
        self.mlp_other = nn.Sequential(
            nn.Linear(7, 128),  # 身高、体重、腰围、臀围、收缩压、舒张压等
            nn.ReLU(inplace=True),
            nn.Linear(128, 512)
        )
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(imagery_dim + 512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, imagery_features, physiological_features):
        """
        前向传播
        
        Args:
            imagery_features (torch.Tensor): 图像特征 (B, 1536)
            physiological_features (torch.Tensor): 生理指标 (B, 8)
                                                   [性别, 年龄, 身高, 体重, 腰围, 臀围, 收缩压, 舒张压]
        
        Returns:
            torch.Tensor: 预测输出 (B, 2)
        """
        # 分割生理指标
        sex_age = physiological_features[:, :2]  # (B, 2)
        other_phys = physiological_features[:, 2:]  # (B, 6)
        
        # 处理性别和年龄（用于调整图像特征）
        sex_age_features = self.mlp_sex_age(sex_age)  # (B, imagery_dim)
        
        # 使用sigmoid调整图像特征的通道权重
        channel_weight = torch.sigmoid(sex_age_features)  # (B, imagery_dim)
        
        # 调整图像特征
        adjusted_imagery = imagery_features * channel_weight  # (B, imagery_dim)
        
        # 处理其他生理指标
        other_features = self.mlp_other(other_phys)  # (B, 512)
        
        # 融合特征
        fused_features = torch.cat([adjusted_imagery, other_features], dim=1)  # (B, imagery_dim+512)
        
        # 分类
        output = self.classifier(fused_features)  # (B, 2)
        
        return output


class FLDDiagnosisModel(nn.Module):
    """
    脂肪肝诊断完整模型（MFF-TDF）
    """
    def __init__(self, num_classes=2, attention_ratio=1/3):
        super(FLDDiagnosisModel, self).__init__()
        
        self.tipnet = TIPNet(attention_ratio=attention_ratio)
        self.mffnet = MFFNet(imagery_dim=1536, phys_dim=8, num_classes=num_classes)
    
    def forward(self, images, physiological_features):
        """
        前向传播
        
        Args:
            images (torch.Tensor): 舌诊图像 (B, 3, 448, 448)
            physiological_features (torch.Tensor): 生理指标 (B, 8)
        
        Returns:
            torch.Tensor: 预测输出 (B, 2)
        """
        # 提取图像特征
        imagery_features = self.tipnet(images)
        
        # 多源特征融合和分类
        output = self.mffnet(imagery_features, physiological_features)
        
        return output
