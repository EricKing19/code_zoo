# 逐日到旬平均温度预测神经网络

这是一个专门设计用于将逐日温度预测转换为旬平均温度预测的深度学习模型。该模型具有简单高效的特点，适用于气象模式数据的后处理。

## 📋 项目概述

### 问题定义
- **输入**: 逐日温度数据（例如30天的逐日温度预报）
- **输出**: 旬平均温度（每10天为一旬，30天数据输出3个旬平均值）
- **应用场景**: 气象数值模式的逐日温度预报后处理

### 模型特点
- ✅ **简单高效**: 轻量级网络架构，训练快速
- ✅ **端到端**: 直接从逐日数据预测旬平均，无需手工特征工程
- ✅ **注意力机制**: 自动学习重要时间步的权重
- ✅ **时间卷积**: 有效捕捉温度的时间序列特征
- ✅ **泛化能力强**: 支持不同长度的输入序列

## 🏗️ 网络架构

### 整体设计
```
逐日温度序列 → 时间卷积网络 → 注意力机制 → 全连接层 → 旬平均温度
    (30天)         (特征提取)     (权重分配)   (非线性映射)    (3个旬)
```

### 详细架构

1. **时间卷积网络 (TCN)**
   - 多层扩张卷积 (Dilated Convolution)
   - 感受野逐层增大，有效捕捉长期依赖
   - 残差连接防止梯度消失

2. **注意力机制**
   - 学习不同日期的重要性权重
   - 突出对旬平均贡献更大的时间点
   - 提高模型的可解释性

3. **特征融合**
   - 全局特征 + 局部加权特征
   - 充分利用时间序列的多尺度信息

4. **输出层**
   - 多层感知机 (MLP)
   - Dropout正则化防止过拟合
   - 直接输出旬平均温度

### 模型参数
- **输入维度**: (batch_size, sequence_length) 例如 (32, 30)
- **隐藏维度**: 64 (可调整)
- **输出维度**: sequence_length // 10，例如30天输出3个旬
- **参数量**: 约25K (轻量级)

## 🚀 快速开始

### 1. 环境安装
```bash
pip install -r requirements.txt
```

### 2. 训练模型
```bash
python daily_to_dekad_temperature_prediction.py
```

### 3. 使用训练好的模型
```bash
python usage_example.py
```

## 📁 文件结构

```
├── daily_to_dekad_temperature_prediction.py  # 主要模型文件
├── usage_example.py                           # 使用示例
├── requirements.txt                          # 依赖包
├── README_temperature_prediction.md         # 说明文档
└── 生成的文件/
    ├── best_model.pth                       # 训练好的模型
    ├── prediction_results.png               # 训练结果可视化
    └── usage_prediction_results.png         # 使用示例结果
```

## 🔧 自定义使用

### 加载您的数据

修改 `usage_example.py` 中的 `load_your_daily_temperature_data()` 函数：

```python
def load_your_daily_temperature_data():
    # 方法1: 从numpy文件加载
    daily_temps = np.load('your_daily_temperature_data.npy')
    
    # 方法2: 从CSV文件加载
    # import pandas as pd
    # df = pd.read_csv('your_temperature_data.csv')
    # daily_temps = df.values
    
    # 方法3: 从其他数据源加载
    # daily_temps = your_data_loading_function()
    
    return daily_temps  # 形状应为 (n_samples, n_days)
```

### 调整模型参数

```python
# 创建预测器时调整参数
predictor = TemperaturePredictor(
    input_days=30,      # 输入天数
    hidden_dim=64,      # 隐藏层维度
    learning_rate=0.001 # 学习率
)
```

### 训练自定义数据

```python
# 准备您的数据
your_daily_data = load_your_data()  # 形状: (n_samples, n_days)

# 创建数据集
dataset = DailyToDekadDataset(your_daily_data)

# 数据分割和训练
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
predictor.train_model(train_loader, val_loader, epochs=100)
```

## 📊 模型性能

### 评估指标
- **RMSE** (均方根误差): 衡量预测的整体准确性
- **MAE** (平均绝对误差): 衡量预测的平均偏差
- **R²** (决定系数): 衡量模型的解释能力

### 典型性能
在合成数据上的测试结果：
- RMSE: < 1.0°C
- MAE: < 0.8°C
- R²: > 0.95

## 🔬 技术细节

### 损失函数
使用均方误差 (MSE) 损失：
```python
loss = MSE(predicted_dekad_temps, true_dekad_temps)
```

### 优化器
AdamW优化器 + 余弦退火学习率调度：
```python
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
```

### 正则化技术
- Dropout (0.1): 防止过拟合
- 梯度裁剪: 防止梯度爆炸
- 早停法: 防止过拟合
- 权重衰减: L2正则化

### 数据处理
- 自动计算旬平均作为监督信号
- 支持批量处理大规模数据
- 内存友好的数据加载

## 💡 使用建议

### 数据准备
1. **数据格式**: 确保输入数据为 (样本数, 天数) 的二维数组
2. **数据质量**: 检查并处理缺失值和异常值
3. **数据范围**: 温度数据应在合理范围内 (-50°C 到 50°C)

### 训练技巧
1. **批量大小**: 根据数据量和内存调整 (推荐32-128)
2. **学习率**: 从0.001开始，根据收敛情况调整
3. **早停**: 设置合适的patience值防止过拟合

### 部署建议
1. **模型保存**: 训练后自动保存最佳模型
2. **批量预测**: 对大量数据使用批量处理
3. **内存管理**: 大数据集分批加载避免内存溢出

## 🔍 扩展功能

### 支持更长时间序列
```python
# 修改输入天数来处理更长序列
predictor = TemperaturePredictor(input_days=60)  # 60天
```

### 多变量输入
可以扩展模型支持多个气象变量：
```python
# 修改模型支持多通道输入 (温度、湿度、气压等)
temporal_conv = TemporalConvNet(input_channels=3, ...)
```

### 空间信息
可以结合卷积神经网络处理格点空间信息：
```python
# 添加2D卷积层处理空间特征
spatial_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
```

## ❓ 常见问题

### Q: 如何处理不同长度的输入序列？
A: 模型支持不同长度输入，只需在初始化时指定 `input_days` 参数。

### Q: 如何提高模型精度？
A: 可以尝试：
- 增加隐藏层维度
- 使用更多训练数据
- 调整学习率和训练轮数
- 添加数据增强

### Q: 模型训练需要多长时间？
A: 在1000个样本上训练通常需要5-10分钟（CPU）或1-2分钟（GPU）。

### Q: 如何解释模型的预测？
A: 模型包含注意力机制，可以可视化不同时间步的权重来理解模型关注的时间点。

## 📚 参考文献

- Temporal Convolutional Networks: [TCN论文](https://arxiv.org/abs/1803.01271)
- Attention Mechanism: [注意力机制](https://arxiv.org/abs/1706.03762)
- 气象预报后处理: 相关气象学文献

## 📧 联系方式

如有问题或建议，请提交issue或联系开发者。

---

**注意**: 这是一个示例实现，实际使用时请根据您的具体数据和需求进行调整。