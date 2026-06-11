# 图像预处理和数据增强模块

import numpy as np
import cv2
from PIL import Image
import random
import torch


class ImagePreprocessor:
    """
    图像预处理类，包括加载、标准化、数据增强等
    """
    
    def __init__(self, image_size=448):
        """
        初始化预处理器
        
        Args:
            image_size (int): 输出图像大小（正方形）
        """
        self.image_size = image_size
    
    def load_image(self, image_path):
        """
        加载图像
        
        Args:
            image_path (str): 图像文件路径
            
        Returns:
            np.ndarray: 加载的图像（RGB格式）
        """
        # 使用PIL加载图像确保RGB格式
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        return image
    
    def resize_image(self, image, size=None):
        """
        调整图像大小
        
        Args:
            image (np.ndarray): 输入图像
            size (int, optional): 目标大小。如果为None，使用self.image_size
            
        Returns:
            np.ndarray: 调整大小后的图像
        """
        if size is None:
            size = self.image_size
        
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        return image
    
    def normalize_image(self, image, mean=None, std=None):
        """
        标准化图像到[0, 1]或使用指定的均值和标准差
        
        Args:
            image (np.ndarray): 输入图像（像素值通常在0-255）
            mean (list, optional): 均值列表（R, G, B）
            std (list, optional): 标准差列表（R, G, B）
            
        Returns:
            np.ndarray: 标准化后的图像
        """
        # 转换为浮点型
        image = image.astype(np.float32)
        
        if mean is not None and std is not None:
            # ImageNet标准化
            image = image / 255.0
            image = (image - np.array(mean)) / np.array(std)
        else:
            # 简单归一化到[0, 1]
            image = image / 255.0
        
        return image
    
    def random_rotation(self, image, angle_range=15):
        """
        随机旋转
        
        Args:
            image (np.ndarray): 输入图像
            angle_range (int): 旋转角度范围（度）
            
        Returns:
            np.ndarray: 旋转后的图像
        """
        angle = random.uniform(-angle_range, angle_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 获取旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 应用旋转
        image = cv2.warpAffine(image, rotation_matrix, (w, h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))
        
        return image
    
    def random_horizontal_flip(self, image):
        """
        随机水平翻转
        
        Args:
            image (np.ndarray): 输入图像
            
        Returns:
            np.ndarray: 翻转后的图像
        """
        if random.random() > 0.5:
            image = cv2.flip(image, 1)  # 1表示水平翻转
        
        return image
    
    def random_crop(self, image, crop_ratio=0.9):
        """
        随机裁剪
        
        Args:
            image (np.ndarray): 输入图像
            crop_ratio (float): 裁剪比例（相对于原图）
            
        Returns:
            np.ndarray: 裁剪并调整到原始大小的图像
        """
        h, w = image.shape[:2]
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        
        # 随机选择裁剪位置
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        
        # 裁剪
        image = image[top:top+crop_h, left:left+crop_w]
        
        # 调整到原始大小
        image = self.resize_image(image, size=(h, w))
        
        return image
    
    def random_scale(self, image, scale_range=(0.9, 1.1)):
        """
        随机缩放
        
        Args:
            image (np.ndarray): 输入图像
            scale_range (tuple): 缩放范围（最小比例, 最大比例）
            
        Returns:
            np.ndarray: 缩放后的图像
        """
        scale = random.uniform(scale_range[0], scale_range[1])
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 缩放
        image = cv2.resize(image, (new_w, new_h))
        
        # 如果缩小了，用黑色边界填充；如果放大了，裁剪到原始大小
        if scale < 1.0:
            # 创建黑色背景
            padded_image = np.zeros((h, w, image.shape[2]), dtype=image.dtype)
            top = (h - new_h) // 2
            left = (w - new_w) // 2
            padded_image[top:top+new_h, left:left+new_w] = image
            image = padded_image
        else:
            # 裁剪到原始大小
            top = (new_h - h) // 2
            left = (new_w - w) // 2
            image = image[top:top+h, left:left+w]
        
        return image
    
    def augment_image(self, image, augmentation_config):
        """
        应用数据增强
        
        Args:
            image (np.ndarray): 输入图像
            augmentation_config (dict): 增强配置字典，包含：
                - rotation_degree: 旋转角度范围
                - scale_range: 缩放范围
                - crop_ratio: 裁剪比例
                - horizontal_flip: 是否进行水平翻转
                
        Returns:
            np.ndarray: 增强后的图像
        """
        # 旋转
        if 'rotation_degree' in augmentation_config:
            image = self.random_rotation(image, augmentation_config['rotation_degree'])
        
        # 缩放
        if 'scale_range' in augmentation_config:
            image = self.random_scale(image, augmentation_config['scale_range'])
        
        # 裁剪
        if 'crop_ratio' in augmentation_config:
            image = self.random_crop(image, augmentation_config['crop_ratio'])
        
        # 水平翻转
        if augmentation_config.get('horizontal_flip', False):
            image = self.random_horizontal_flip(image)
        
        return image
    
    def preprocess_image(self, image_path, augment=False, augmentation_config=None, 
                        normalize=True, mean=None, std=None):
        """
        完整的预处理流程
        
        Args:
            image_path (str): 图像路径
            augment (bool): 是否进行数据增强
            augmentation_config (dict): 增强配置
            normalize (bool): 是否进行标准化
            mean (list, optional): 标准化均值
            std (list, optional): 标准化标准差
            
        Returns:
            np.ndarray: 预处理后的图像
        """
        # 加载图像
        image = self.load_image(image_path)
        
        # 调整大小
        image = self.resize_image(image)
        
        # 数据增强
        if augment and augmentation_config is not None:
            image = self.augment_image(image, augmentation_config)
        
        # 标准化
        if normalize:
            image = self.normalize_image(image, mean, std)
        
        return image
    
    def preprocess_batch(self, image_paths, augment=False, augmentation_config=None,
                        normalize=True, mean=None, std=None):
        """
        批量预处理图像
        
        Args:
            image_paths (list): 图像路径列表
            augment (bool): 是否进行数据增强
            augmentation_config (dict): 增强配置
            normalize (bool): 是否进行标准化
            mean (list, optional): 标准化均值
            std (list, optional): 标准化标准差
            
        Returns:
            torch.Tensor: 预处理后的图像张量（批次×通道×高×宽）
        """
        images = []
        
        for image_path in image_paths:
            image = self.preprocess_image(
                image_path,
                augment=augment,
                augmentation_config=augmentation_config,
                normalize=normalize,
                mean=mean,
                std=std
            )
            images.append(image)
        
        # 转换为张量并调整维度
        images = np.array(images)  # (N, H, W, C)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()  # (N, C, H, W)
        
        return images
