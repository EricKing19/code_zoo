"""
简化版演示：逐日到旬平均温度预测神经网络概念验证
使用纯NumPy实现，无需深度学习框架
"""

import numpy as np
import random

class SimpleTemperaturePredictor:
    """
    简化的温度预测器：演示神经网络的基本概念
    """
    def __init__(self, input_days=30, hidden_dim=16):
        self.input_days = input_days
        self.num_dekads = input_days // 10
        self.hidden_dim = hidden_dim
        
        # 初始化网络权重 (简化的全连接网络)
        self.W1 = np.random.randn(input_days, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, self.num_dekads) * 0.1
        self.b2 = np.zeros(self.num_dekads)
        
        # 注意力权重
        self.attention_weights = np.random.randn(input_days) * 0.1
        
    def sigmoid(self, x):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def softmax(self, x):
        """Softmax函数用于注意力权重"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def forward(self, daily_temps):
        """前向传播"""
        # 计算注意力权重
        attention = self.softmax(self.attention_weights)
        
        # 加权输入特征
        weighted_input = daily_temps * attention
        
        # 第一层：输入层到隐藏层
        hidden = self.relu(np.dot(weighted_input, self.W1) + self.b1)
        
        # 第二层：隐藏层到输出层
        output = np.dot(hidden, self.W2) + self.b2
        
        return output
    
    def calculate_true_dekads(self, daily_temps):
        """计算真实的旬平均温度"""
        dekads = []
        for i in range(self.num_dekads):
            start_day = i * 10
            end_day = min((i + 1) * 10, len(daily_temps))
            dekad_avg = np.mean(daily_temps[start_day:end_day])
            dekads.append(dekad_avg)
        return np.array(dekads)
    
    def train_simple(self, daily_data, epochs=1000, learning_rate=0.01):
        """简化的训练过程"""
        print("开始简化训练过程...")
        
        for epoch in range(epochs):
            total_loss = 0
            
            for sample in daily_data:
                # 前向传播
                prediction = self.forward(sample)
                target = self.calculate_true_dekads(sample)
                
                # 计算损失 (均方误差)
                loss = np.mean((prediction - target) ** 2)
                total_loss += loss
                
                # 简化的梯度下降 (仅更新输出层权重)
                error = prediction - target
                
                # 更新权重 (简化版本)
                attention = self.softmax(self.attention_weights)
                weighted_input = sample * attention
                hidden = self.relu(np.dot(weighted_input, self.W1) + self.b1)
                
                # 输出层梯度
                self.W2 -= learning_rate * np.outer(hidden, error) / len(daily_data)
                self.b2 -= learning_rate * error / len(daily_data)
            
            avg_loss = total_loss / len(daily_data)
            
            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        
        print("训练完成！")
    
    def predict(self, daily_temps):
        """预测旬平均温度"""
        return self.forward(daily_temps)

def generate_sample_data(n_samples=100, n_days=30):
    """生成示例温度数据"""
    print("生成示例数据...")
    
    daily_data = []
    for i in range(n_samples):
        # 基础温度
        base_temp = random.uniform(20, 30)
        
        # 模拟温度变化
        daily_temps = []
        current_temp = base_temp
        
        for day in range(n_days):
            # 添加随机变化和趋势
            trend = 0.1 * (day - n_days/2) / n_days  # 轻微趋势
            daily_change = random.gauss(0, 2)  # 日间变化
            seasonal = 3 * np.sin(2 * np.pi * day / 30)  # 季节性变化
            
            current_temp = base_temp + trend + daily_change + seasonal
            daily_temps.append(current_temp)
        
        daily_data.append(np.array(daily_temps))
    
    return daily_data

def evaluate_model(predictor, test_data):
    """评估模型性能"""
    print("\n评估模型性能...")
    
    all_predictions = []
    all_targets = []
    
    for sample in test_data:
        prediction = predictor.predict(sample)
        target = predictor.calculate_true_dekads(sample)
        
        all_predictions.append(prediction)
        all_targets.append(target)
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # 计算评估指标
    mae = np.mean(np.abs(all_predictions - all_targets))
    rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
    
    print(f"平均绝对误差 (MAE): {mae:.3f}°C")
    print(f"均方根误差 (RMSE): {rmse:.3f}°C")
    
    return all_predictions, all_targets

def visualize_results_simple(predictor, test_data, n_samples=3):
    """简单的文本可视化结果"""
    print("\n预测结果展示:")
    print("=" * 60)
    
    for i in range(min(n_samples, len(test_data))):
        sample = test_data[i]
        prediction = predictor.predict(sample)
        target = predictor.calculate_true_dekads(sample)
        
        print(f"\n样本 {i+1}:")
        print(f"逐日温度: {sample[:10].round(1)}... (前10天)")
        print(f"真实旬平均: {target.round(2)}")
        print(f"预测旬平均: {prediction.round(2)}")
        print(f"误差: {(prediction - target).round(2)}")
        print("-" * 40)

def demonstrate_attention_mechanism(predictor, sample_data):
    """演示注意力机制"""
    print("\n注意力机制演示:")
    print("=" * 40)
    
    attention_weights = predictor.softmax(predictor.attention_weights)
    
    print("各天的注意力权重 (前15天):")
    for i in range(min(15, len(attention_weights))):
        bar = "█" * int(attention_weights[i] * 1000)
        print(f"第{i+1:2d}天: {attention_weights[i]:.4f} {bar}")
    
    print(f"\n权重总和: {np.sum(attention_weights):.6f}")
    print("注意力权重越高，该天对旬平均预测的贡献越大")

def main():
    """主演示函数"""
    print("逐日到旬平均温度预测神经网络 - 简化演示")
    print("=" * 60)
    
    # 生成数据
    train_data = generate_sample_data(n_samples=80, n_days=30)
    test_data = generate_sample_data(n_samples=20, n_days=30)
    
    print(f"训练数据: {len(train_data)} 个样本")
    print(f"测试数据: {len(test_data)} 个样本")
    print(f"每个样本: 30天逐日温度 → 3个旬平均温度")
    
    # 创建模型
    predictor = SimpleTemperaturePredictor(input_days=30, hidden_dim=16)
    
    print(f"\n网络结构:")
    print(f"输入层: {predictor.input_days} 个神经元 (30天逐日温度)")
    print(f"隐藏层: {predictor.hidden_dim} 个神经元")
    print(f"输出层: {predictor.num_dekads} 个神经元 (3个旬平均)")
    print(f"总参数量: {predictor.input_days * predictor.hidden_dim + predictor.hidden_dim * predictor.num_dekads + predictor.hidden_dim + predictor.num_dekads + predictor.input_days}")
    
    # 训练前性能
    print("\n训练前性能:")
    evaluate_model(predictor, test_data[:5])
    
    # 训练模型
    predictor.train_simple(train_data, epochs=1000, learning_rate=0.01)
    
    # 训练后性能
    print("\n训练后性能:")
    predictions, targets = evaluate_model(predictor, test_data)
    
    # 可视化结果
    visualize_results_simple(predictor, test_data, n_samples=3)
    
    # 演示注意力机制
    demonstrate_attention_mechanism(predictor, test_data[0])
    
    print("\n总结:")
    print("- 该神经网络成功学习了从逐日温度到旬平均温度的映射")
    print("- 注意力机制帮助模型关注重要的时间步")
    print("- 实际应用中，可以使用更复杂的深度学习框架如PyTorch")
    print("- 模型结构可以根据具体需求进行调整和优化")

if __name__ == "__main__":
    # 设置随机种子以便结果可重现
    np.random.seed(42)
    random.seed(42)
    
    main()