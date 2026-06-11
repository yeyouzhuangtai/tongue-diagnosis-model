# 数据集类模块

import torch
from torch.utils.data import Dataset
import numpy as np
from src.preprocessing import ImagePreprocessor


class TongueImageDataset(Dataset):
    """
    舌诊图像数据集类
    """
    
    def __init__(self, image_paths, physiological_features, labels, 
                 augment=False, augmentation_config=None):
        """
        初始化数据集
        
        Args:
            image_paths (list): 图像路径列表
            physiological_features (np.ndarray): 生理指标特征矩阵 (N, 8)
            labels (list or np.ndarray): 标签列表
            augment (bool): 是否进行数据增强
            augmentation_config (dict): 数据增强配置
        """
        self.image_paths = image_paths
        self.physiological_features = physiological_features
        self.labels = np.array(labels)
        self.augment = augment
        self.augmentation_config = augmentation_config or {}
        
        # 初始化预处理器
        self.preprocessor = ImagePreprocessor(image_size=448)
        
        assert len(image_paths) == len(physiological_features) == len(labels), \
            "图像、生理特征和标签数量不一致"
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple: (image_tensor, physiological_features, label)
        """
        # 加载和预处理图像
        image = self.preprocessor.preprocess_image(
            self.image_paths[idx],
            augment=self.augment,
            augmentation_config=self.augmentation_config,
            normalize=True
        )
        
        # 转换为张量
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # (C, H, W)
        
        # 获取生理指标
        phys_features = torch.from_numpy(self.physiological_features[idx]).float()
        
        # 获取标签
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image_tensor, phys_features, label
