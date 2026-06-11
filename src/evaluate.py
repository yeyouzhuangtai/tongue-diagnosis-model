# 评估和可视化模块

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc
)
import os


class Evaluator:
    """
    模型评估类
    """
    
    def __init__(self, model, device, output_dir='./output'):
        """
        初始化评估器
        
        Args:
            model: PyTorch模型
            device: 计算设备
            output_dir: 输出目录
        """
        self.model = model
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate(self, val_loader):
        """
        评估模型性能
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            dict: 评估指标
        """
        self.model.eval()
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for images, phys_features, labels in val_loader:
                images = images.to(self.device)
                phys_features = phys_features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images, phys_features)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        auc_score = roc_auc_score(all_labels, all_probs)
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'probabilities': all_probs,
            'labels': all_labels
        }
        
        return metrics
    
    def plot_roc_curve(self, metrics, filename='roc_curve.png'):
        """
        绘制ROC曲线
        
        Args:
            metrics (dict): 评估指标
            filename (str): 保存文件名
        """
        fpr, tpr, _ = roc_curve(metrics['labels'], metrics['probabilities'])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved to {filepath}")
    
    def plot_confusion_matrix(self, metrics, filename='confusion_matrix.png'):
        """
        绘制混淆矩阵
        
        Args:
            metrics (dict): 评估指标
            filename (str): 保存文件名
        """
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Non-FLD', 'FLD'])
        plt.yticks(tick_marks, ['Non-FLD', 'FLD'])
        
        # 添加文本
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {filepath}")
    
    def plot_training_history(self, history, filename='training_history.png'):
        """
        绘制训练历史
        
        Args:
            history (dict): 训练历史
            filename (str): 保存文件名
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 损失
        axes[0].plot(history['train_loss'], label='Training Loss', marker='o')
        axes[0].plot(history['val_loss'], label='Validation Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 指标
        if history['val_f1']:
            axes[1].plot(history['val_f1'], label='F1-Score', marker='o')
            axes[1].plot(history['val_auc'], label='AUC', marker='s')
            axes[1].set_xlabel('Validation Step')
            axes[1].set_ylabel('Score')
            axes[1].set_title('Validation Metrics')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training history saved to {filepath}")
    
    def print_metrics_summary(self, metrics):
        """
        打印指标总结
        
        Args:
            metrics (dict): 评估指标
        """
        print("\n" + "="*50)
        print("模型性能指标")
        print("="*50)
        print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"精准率 (Precision): {metrics['precision']:.4f}")
        print(f"召回率 (Recall): {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print("="*50 + "\n")
    
    def save_metrics_to_file(self, metrics, filename='metrics.txt'):
        """
        将指标保存到文件
        
        Args:
            metrics (dict): 评估指标
            filename (str): 文件名
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("脂肪肝舌诊预测模型 - 性能指标\n")
            f.write("=" * 50 + "\n")
            f.write(f"准确率 (Accuracy): {metrics['accuracy']:.4f}\n")
            f.write(f"精准率 (Precision): {metrics['precision']:.4f}\n")
            f.write(f"召回率 (Recall): {metrics['recall']:.4f}\n")
            f.write(f"F1-Score: {metrics['f1']:.4f}\n")
            f.write(f"AUC: {metrics['auc']:.4f}\n")
            f.write("=" * 50 + "\n")
            f.write("\n混淆矩阵:\n")
            f.write(str(metrics['confusion_matrix']))
        
        print(f"Metrics saved to {filepath}")
