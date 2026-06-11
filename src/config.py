# 配置文件 - 集中管理所有超参数

import os
import torch

# ==================== 数据路径配置 ====================
# 根据你的实际情况修改这些路径
DATA_ROOT = r"D:\要用的\数据挖掘林光毅大作业"  # 数据集根目录
IMAGE_FOLDER = os.path.join(DATA_ROOT, "images")  # 舌诊图片文件夹
LABEL_FILE = os.path.join(DATA_ROOT, "labels.csv")  # 标签文件

# 输出路径
OUTPUT_DIR = "./output"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
RESULTS_SAVE_PATH = os.path.join(OUTPUT_DIR, "results")

# ==================== 模型参数配置 ====================
# 图像输入大小
IMAGE_SIZE = 448

# TIPNet参数
TIPNET_INPUT_CHANNELS = 3
TIPNET_STEM_CHANNELS = 192
TIPNET_BRANCH_CHANNELS = 512
TIPNET_OUTPUT_CHANNELS = 1536  # 三个分支拼接: 512*3
TIPNET_ATTENTION_RATIO = 1/3  # 通道注意和空间注意的平衡参数

# MFFNet参数
PHYSIOLOGICAL_FEATURES_DIM = 8  # 性别、年龄、身高、腰围、臀围、体重、收缩压、舒张压
IMAGERY_FEATURE_DIM = 1536
FUSION_OUTPUT_DIM = 2  # 二分类：有脂肪肝/无脂肪肝

# ==================== 训练参数配置 ====================
# 训练轮数
EPOCHS = 150
WARM_UP_EPOCHS = 5

# Batch size
BATCH_SIZE = 64

# 学习率
INITIAL_LR = 0.001
WEIGHT_DECAY = 0.0001

# 余弦退火调度参数
LR_MIN = 1e-6
LR_MAX = INITIAL_LR

# 加权交叉熵损失的权重系数
LOSS_BALANCE_COEFFICIENT = 0.9  # β参数，用于处理类别不平衡

# ==================== 数据集参数配置 ====================
# 训练集/验证集分割比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2

# 是否进行五折交叉验证
USE_K_FOLD = False
K_FOLDS = 5

# ==================== 数据增强参数配置 ====================
# 旋转角度范围（度）
ROTATION_DEGREE = 15

# 缩放范围
SCALE_RANGE = (0.9, 1.1)

# 裁剪比例
CROP_RATIO = 0.9

# 是否进行水平翻转
HORIZONTAL_FLIP = True

# ==================== 设备配置 ====================
# 自动选择GPU或CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 其他配置 ====================
# 随机种子（保证可重复性）
RANDOM_SEED = 42

# 是否在控制台打印详细日志
VERBOSE = True

# 验证频率（每多少个epoch进行一次验证）
VAL_INTERVAL = 1

# 保存最优模型
SAVE_BEST_MODEL = True
