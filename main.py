# 主程序入口

import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np

# 导入配置和模块
from src.config import *
from src.utils import (
    set_random_seed, create_directories, setup_logger,
    load_image_paths_and_labels, print_model_info
)
from src.preprocessing import ImagePreprocessor
from src.dataset import TongueImageDataset
from src.models import FLDDiagnosisModel
from src.train import Trainer
from src.evaluate import Evaluator


def main():
    """
    主程序
    """
    # 设置随机种子
    set_random_seed(RANDOM_SEED)
    
    # 创建输出目录
    create_directories([OUTPUT_DIR, RESULTS_SAVE_PATH])
    
    # 设置日志
    logger = setup_logger(
        'FLD_Diagnosis',
        log_file=os.path.join(OUTPUT_DIR, 'training.log')
    )
    
    logger.info("="*50)
    logger.info("脂肪肝舌诊预测模型 - 开始训练")
    logger.info("="*50)
    logger.info(f"使用设备: {DEVICE}")
    logger.info(f"数据路径: {DATA_ROOT}")
    
    # ==================== 数据加载 ====================
    logger.info("\n正在加载数据...")
    
    try:
        image_paths, labels, ids = load_image_paths_and_labels(IMAGE_FOLDER, LABEL_FILE)
        logger.info(f"成功加载 {len(image_paths)} 个样本")
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return
    
    # 检查数据
    if len(image_paths) == 0:
        logger.error("未找到任何图像文件，请检查数据路径")
        return
    
    # 加载生理指标
    logger.info("正在加载生理指标...")
    import pandas as pd
    try:
        df = pd.read_csv(LABEL_FILE)
        
        # 提取生理指标
        phys_columns = ['sex', 'age', 'height', 'waist', 'hip', 'weight', 'sbp', 'dbp']
        
        available_columns = [col for col in phys_columns if col in df.columns]
        if len(available_columns) < 8:
            logger.warning(f"CSV文件中缺少某些列。找到的列: {available_columns}")
            phys_columns = available_columns
        
        # 创建生理指标数据
        physiological_features_list = []
        for image_path in image_paths:
            filename = os.path.basename(image_path)
            sample_id = os.path.splitext(filename)[0]
            
            row = df[df['id'] == int(sample_id)]
            if len(row) > 0:
                phys_features = row[phys_columns].values[0].astype(float)
                physiological_features_list.append(phys_features)
            else:
                logger.warning(f"未找到ID {sample_id} 的生理指标")
                physiological_features_list.append(np.zeros(len(phys_columns)))
        
        physiological_features = np.array(physiological_features_list)
        
        # 标准化生理指标
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        physiological_features = scaler.fit_transform(physiological_features)
        
        logger.info(f"生理指标形状: {physiological_features.shape}")
        
    except Exception as e:
        logger.error(f"生理指标加载失败: {e}")
        return
    
    # 类别分布
    unique, counts = np.unique(labels, return_counts=True)
    logger.info(f"类别分布 - 无脂肪肝: {counts[0]}, 有脂肪肝: {counts[1]}")
    
    # ==================== 数据集创建 ====================
    logger.info("\n正在创建数据集...")
    
    full_dataset = TongueImageDataset(
        image_paths,
        physiological_features,
        labels,
        augment=False
    )
    
    # 分割训练集和验证集
    train_size = int(len(full_dataset) * TRAIN_RATIO)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    
    # 为训练集启用数据增强
    augmentation_config = {
        'rotation_degree': ROTATION_DEGREE,
        'scale_range': SCALE_RANGE,
        'crop_ratio': CROP_RATIO,
        'horizontal_flip': HORIZONTAL_FLIP
    }
    train_dataset.dataset.augment = True
    train_dataset.dataset.augmentation_config = augmentation_config
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # ==================== 模型初始化 ====================
    logger.info("\n初始化模型...")
    
    model = FLDDiagnosisModel(
        num_classes=2,
        attention_ratio=TIPNET_ATTENTION_RATIO
    )
    
    print_model_info(model, logger)
    
    # ==================== 训练 ====================
    logger.info("\n开始训练...")
    
    trainer = Trainer(model, DEVICE, config=globals(), logger=logger)
    
    history = trainer.train(
        train_loader,
        val_loader,
        checkpoint_dir=MODEL_SAVE_PATH
    )
    
    # ==================== 评估 ====================
    logger.info("\n评估模型...")
    
    evaluator = Evaluator(model, DEVICE, output_dir=RESULTS_SAVE_PATH)
    metrics = evaluator.evaluate(val_loader)
    
    # 打印指标
    evaluator.print_metrics_summary(metrics)
    
    # 保存指标到文件
    evaluator.save_metrics_to_file(metrics)
    
    # 生成可视化
    logger.info("\n生成可视化图表...")
    evaluator.plot_roc_curve(metrics)
    evaluator.plot_confusion_matrix(metrics)
    evaluator.plot_training_history(history)
    
    logger.info("\n" + "="*50)
    logger.info("训练完成！")
    logger.info(f"结果保存路径: {RESULTS_SAVE_PATH}")
    logger.info("="*50)


if __name__ == "__main__":
    main()
