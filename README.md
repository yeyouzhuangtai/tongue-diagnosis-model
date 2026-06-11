# 脂肪肝舌诊预测模型

## 项目简介

本项目复现了学术论文 "Accurate fatty liver disease diagnosis with a multi-source feature fusion model on the segmented tongue image dataset" 中提出的MFF-TDF框架，建立了一个基于舌诊图像和生理指标的脂肪肝预测分类模型。

## 项目特点

✅ **多模态融合** - 联合利用舌诊图像和生理指标（性别、年龄、身高、体重等）  
✅ **先进网络架构** - 采用多支路多尺度TIPNet进行特征提取  
✅ **注意力机制** - 通道和空间注意力机制增强特征  
✅ **PyCharm友好** - 完整的项目结构，适合本地实践  
✅ **完整评估** - AUC、F1-score、精准率、召回率等多项指标  
✅ **可视化结果** - ROC曲线、混淆矩阵、训练曲线  

## 项目结构

```
tongue-diagnosis-model/
├── data/
│   ├── images/              # 舌诊图片文件夹
│   └── labels.csv          # 标签文件
├── src/
│   ├── __init__.py
│   ├── config.py           # 配置文件
│   ├── preprocessing.py    # 图像预处理
│   ├── dataset.py          # 数据集类
│   ├── models.py           # 模型定义
│   ├── train.py            # 训练脚本
│   ├── evaluate.py         # 评估脚本
│   └── utils.py            # 工具函数
├── output/                 # 输出目录
├── main.py                 # 主程序
├── requirements.txt        # 依赖包
└── README.md              # 说明文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将数据放在 `D:\要用的\数据挖掘林光毅大作业` 下：

**目录结构：**
```
D:\要用的\数据挖掘林光毅大作业\
├── images/    # 舌诊图片
└── labels.csv # 标签文件
```

**CSV格式：**
```
id,sex,age,height,waist,hip,weight,sbp,dbp,FLD
1,0,45,170,85,95,75,120,80,0
2,1,52,165,90,100,68,130,85,1
```

### 3. 运行训练

```bash
python main.py
```

### 4. 查看结果

结果保存在 `output/results/` 目录：
- `metrics.txt` - 性能指标
- `roc_curve.png` - ROC曲线
- `confusion_matrix.png` - 混淆矩阵
- `training_history.png` - 训练曲线

## 模型架构

### TIPNet（舌图特征感知网络）
- **多支路多尺度卷积**：3×3、5×5、7×7并行处理
- **舌特征注意力残差块**：TCB-Block和TCN-Block
- **注意力机制**：通道和空间注意力融合
- **输出**：1536维特征向量

### MFFNet（多源特征融合网络）
- 性别/年龄：调整图像特征权重
- 其他指标：直接与图像特征拼接
- 最终分类器：输出FLD预测

## 评估指标

- **Accuracy（准确率）** - 正确预测比例
- **Precision（精准率）** - TP/(TP+FP)
- **Recall（召回率）** - TP/(TP+FN)
- **F1-score** - 精准率和召回率的调和平均
- **AUC** - ROC曲线下面积

## 论文对标

论文最优性能：
- F1-score: 0.797
- AUC: 0.924
- Recall: 0.847

## 常见问题

**Q: 找不到图像文件？**  
A: 检查数据路径是否正确，确保图片在 `data/images/` 目录下

**Q: 显存不足？**  
A: 在 `src/config.py` 中减小 `BATCH_SIZE`

**Q: 如何修改超参数？**  
A: 编辑 `src/config.py` 文件，无需改动其他代码

## 依赖包

- torch >= 2.0.1
- torchvision >= 0.15.2
- numpy >= 1.24.3
- pandas >= 2.0.3
- scikit-learn >= 1.3.0
- opencv-python >= 4.8.0.74
- matplotlib >= 3.7.2

## 许可证

MIT
