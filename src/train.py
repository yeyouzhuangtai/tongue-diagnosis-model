# 训练模块

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from src.utils import setup_logger, save_checkpoint


class Trainer:
    """
    模型训练器
    """
    
    def __init__(self, model, device, config, logger=None):
        """
        初始化训练器
        
        Args:
            model: PyTorch模型
            device: 计算设备
            config: 配置对象
            logger: 日志记录器
        """
        self.model = model
        self.device = device
        self.config = config
        self.logger = logger
        
        # 移动模型到设备
        self.model.to(device)
        
        # 初始化优化器
        self.optimizer = SGD(
            self.model.parameters(),
            lr=config.INITIAL_LR,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 学习率调度器（余弦退火）
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.EPOCHS - config.WARM_UP_EPOCHS,
            eta_min=config.LR_MIN
        )
        
        # 损失函数（加权交叉熵）
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.best_metrics = {'val_loss': float('inf'), 'val_f1': 0.0, 'val_auc': 0.0}
    
    def train_epoch(self, train_loader, epoch):
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch数
            
        Returns:
            dict: 训练指标
        """
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS}", disable=not self.config.VERBOSE)
        
        for batch_idx, (images, phys_features, labels) in enumerate(progress_bar):
            # 移动数据到设备
            images = images.to(self.device)
            phys_features = phys_features.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images, phys_features)
            
            # 计算损失
            loss = self.loss_fn(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            
            # 记录预测和标签
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        
        # 计算训练准确率
        train_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        metrics = {
            'train_loss': avg_loss,
            'train_accuracy': train_accuracy
        }
        
        if self.logger:
            self.logger.info(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        
        return metrics
    
    def validate(self, val_loader):
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            dict: 验证指标
        """
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for images, phys_features, labels in val_loader:
                # 移动数据到设备
                images = images.to(self.device)
                phys_features = phys_features.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                outputs = self.model(images, phys_features)
                
                # 计算损失
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                
                # 获取预测和概率
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # 正类概率
                all_labels.extend(labels.cpu().numpy())
        
        # 计算平均损失
        avg_loss = total_loss / len(val_loader)
        
        # 计算验证指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        auc = roc_auc_score(all_labels, all_probs)
        
        metrics = {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'val_auc': auc
        }
        
        if self.logger:
            self.logger.info(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        return metrics
    
    def train(self, train_loader, val_loader, checkpoint_dir=None):
        """
        完整的训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            checkpoint_dir: 检查点保存目录
            
        Returns:
            dict: 训练历史
        """
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'val_auc': []
        }
        
        for epoch in range(self.config.EPOCHS):
            # 预热学习率
            if epoch < self.config.WARM_UP_EPOCHS:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.INITIAL_LR * (epoch + 1) / self.config.WARM_UP_EPOCHS
            else:
                self.scheduler.step()
            
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_accuracy'].append(train_metrics['train_accuracy'])
            
            # 验证
            if (epoch + 1) % self.config.VAL_INTERVAL == 0:
                val_metrics = self.validate(val_loader)
                history['val_loss'].append(val_metrics['val_loss'])
                history['val_accuracy'].append(val_metrics['val_accuracy'])
                history['val_f1'].append(val_metrics['val_f1'])
                history['val_auc'].append(val_metrics['val_auc'])
                
                # 保存最优模型
                if self.config.SAVE_BEST_MODEL:
                    # 基于F1-score保存最优模型
                    if val_metrics['val_f1'] > self.best_metrics['val_f1']:
                        self.best_metrics = val_metrics
                        if checkpoint_dir:
                            save_checkpoint(
                                self.model,
                                self.optimizer,
                                epoch,
                                self.best_metrics,
                                checkpoint_dir
                            )
                            if self.logger:
                                self.logger.info(f"Best model saved at epoch {epoch+1}")
        
        return history
