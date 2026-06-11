# 工具函数模块

import os
import random
import numpy as np
import torch
import logging
from pathlib import Path


def set_random_seed(seed):
    """
    设置随机种子以保证可重复性
    
    Args:
        seed (int): 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 确保使用确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(paths):
    """
    创建多个目录
    
    Args:
        paths (list or str): 目录路径或路径列表
    """
    if isinstance(paths, str):
        paths = [paths]
    
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    设置日志记录器
    
    Args:
        name (str): 日志记录器名称
        log_file (str, optional): 日志文件路径
        level: 日志级别
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 创建日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_image_paths_and_labels(image_folder, label_file):
    """
    从文件夹和CSV标签文件加载图像路径和标签
    
    Args:
        image_folder (str): 图像文件夹路径
        label_file (str): CSV标签文件路径
        
    Returns:
        tuple: (image_paths, labels, ids) - 图像路径列表、标签列表、样本ID列表
    """
    import pandas as pd
    
    # 读取CSV标签文件
    df = pd.read_csv(label_file)
    
    # 假设CSV中有'id'和'FLD'两列
    # 如果列名不同，请修改下面的列名
    if 'id' not in df.columns or 'FLD' not in df.columns:
        raise ValueError(f"CSV文件必须包含'id'和'FLD'列。实际列名：{df.columns.tolist()}")
    
    image_paths = []
    labels = []
    ids = []
    
    for idx, row in df.iterrows():
        sample_id = str(row['id'])
        label = int(row['FLD'])
        
        # 尝试找到对应的图像文件（支持常见的图像格式）
        image_found = False
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
            image_path = os.path.join(image_folder, f"{sample_id}{ext}")
            if os.path.exists(image_path):
                image_paths.append(image_path)
                labels.append(label)
                ids.append(sample_id)
                image_found = True
                break
        
        if not image_found:
            print(f"警告：找不到ID为{sample_id}的图像文件")
    
    return image_paths, labels, ids


def get_class_weights(labels, beta=0.9):
    """
    计算加权交叉熵的类权重
    
    Args:
        labels (list): 标签列表
        beta (float): 平衡系数，范围[0, 1]，β越小权重差异越大
        
    Returns:
        torch.Tensor: 类权重张量
    """
    unique_labels = np.unique(labels)
    weights = []
    
    total_samples = len(labels)
    
    for label in unique_labels:
        class_count = np.sum(np.array(labels) == label)
        # 加权公式：w_i = (1 - β) / (1 - β^n_i)，其中n_i是类i的样本数
        weight = (1 - beta) / (1 - beta ** class_count)
        weights.append(weight)
    
    weights = torch.tensor(weights, dtype=torch.float32)
    # 归一化权重
    weights = weights / weights.sum()
    
    return weights


def print_model_info(model, logger=None):
    """
    打印模型信息（参数数量等）
    
    Args:
        model: PyTorch模型
        logger: 日志记录器（可选）
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info_str = f"\n{'='*50}\n模型信息\n{'='*50}\n"
    info_str += f"总参数数：{total_params:,}\n"
    info_str += f"可训练参数数：{trainable_params:,}\n"
    info_str += f"{'='*50}\n"
    
    if logger:
        logger.info(info_str)
    else:
        print(info_str)


def save_checkpoint(model, optimizer, epoch, best_metrics, save_path):
    """
    保存模型检查点
    
    Args:
        model: PyTorch模型
        optimizer: 优化器
        epoch (int): 当前epoch
        best_metrics (dict): 最佳性能指标
        save_path (str): 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metrics': best_metrics
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(model, optimizer, load_path):
    """
    加载模型检查点
    
    Args:
        model: PyTorch模型
        optimizer: 优化器
        load_path (str): 加载路径
        
    Returns:
        tuple: (epoch, best_metrics)
    """
    checkpoint = torch.load(load_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_metrics = checkpoint['best_metrics']
    
    return epoch, best_metrics
