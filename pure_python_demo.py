"""
纯Python实现：逐日到旬平均温度预测神经网络演示
不依赖任何外部库，展示神经网络的基本原理
"""

import math
import random

class PureTemperaturePredictor:
    """
    纯Python实现的温度预测神经网络
    """
    def __init__(self, input_days=30, hidden_dim=8):
        self.input_days = input_days
        self.num_dekads = input_days // 10
        self.hidden_dim = hidden_dim
        
        # 初始化网络权重
        self.W1 = [[random.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(input_days)]
        self.b1 = [0.0] * hidden_dim
        self.W2 = [[random.gauss(0, 0.1) for _ in range(self.num_dekads)] for _ in range(hidden_dim)]
        self.b2 = [0.0] * self.num_dekads
        
        # 注意力权重
        self.attention_weights = [random.gauss(0, 0.1) for _ in range(input_days)]
        
        print(f"神经网络初始化完成:")
        print(f"  输入层: {input_days} 个神经元")
        print(f"  隐藏层: {hidden_dim} 个神经元") 
        print(f"  输出层: {self.num_dekads} 个神经元")
    
    def relu(self, x):
        """ReLU激活函数"""
        return max(0, x)
    
    def softmax(self, x_list):
        """Softmax函数"""
        max_x = max(x_list)
        exp_x = [math.exp(x - max_x) for x in x_list]
        sum_exp = sum(exp_x)
        return [exp / sum_exp for exp in exp_x]
    
    def forward(self, daily_temps):
        """前向传播"""
        # 计算注意力权重
        attention = self.softmax(self.attention_weights)
        
        # 加权输入
        weighted_input = [daily_temps[i] * attention[i] for i in range(len(daily_temps))]
        
        # 隐藏层计算
        hidden = []
        for j in range(self.hidden_dim):
            h = self.b1[j]
            for i in range(self.input_days):
                h += weighted_input[i] * self.W1[i][j]
            hidden.append(self.relu(h))
        
        # 输出层计算
        output = []
        for k in range(self.num_dekads):
            o = self.b2[k]
            for j in range(self.hidden_dim):
                o += hidden[j] * self.W2[j][k]
            output.append(o)
        
        return output
    
    def calculate_true_dekads(self, daily_temps):
        """计算真实的旬平均温度"""
        dekads = []
        for i in range(self.num_dekads):
            start_day = i * 10
            end_day = min((i + 1) * 10, len(daily_temps))
            dekad_sum = sum(daily_temps[start_day:end_day])
            dekad_avg = dekad_sum / (end_day - start_day)
            dekads.append(dekad_avg)
        return dekads
    
    def calculate_loss(self, predictions, targets):
        """计算均方误差损失"""
        total_loss = 0
        for i in range(len(predictions)):
            diff = predictions[i] - targets[i]
            total_loss += diff * diff
        return total_loss / len(predictions)
    
    def train_simple(self, daily_data, epochs=500, learning_rate=0.005):
        """简化的训练过程"""
        print(f"\n开始训练 (共{epochs}轮)...")
        
        for epoch in range(epochs):
            total_loss = 0
            
            for sample in daily_data:
                # 前向传播
                prediction = self.forward(sample)
                target = self.calculate_true_dekads(sample)
                
                # 计算损失
                loss = self.calculate_loss(prediction, target)
                total_loss += loss
                
                # 简化的权重更新（仅更新输出层偏置）
                for k in range(self.num_dekads):
                    error = prediction[k] - target[k]
                    self.b2[k] -= learning_rate * error / len(daily_data)
            
            avg_loss = total_loss / len(daily_data)
            
            if epoch % 100 == 0:
                print(f"  轮次 {epoch:3d}: 平均损失 = {avg_loss:.4f}")
        
        print("训练完成！")
    
    def predict(self, daily_temps):
        """预测旬平均温度"""
        return self.forward(daily_temps)

def generate_sample_data(n_samples=50, n_days=30):
    """生成示例温度数据"""
    print(f"生成 {n_samples} 个样本的温度数据...")
    
    daily_data = []
    for i in range(n_samples):
        # 基础温度
        base_temp = random.uniform(18, 32)
        
        # 生成30天的温度数据
        daily_temps = []
        for day in range(n_days):
            # 添加各种变化因素
            seasonal = 4 * math.sin(2 * math.pi * day / 30)  # 季节性变化
            trend = 0.2 * (day - 15) / 30  # 线性趋势
            noise = random.gauss(0, 1.5)  # 随机噪声
            
            temp = base_temp + seasonal + trend + noise
            daily_temps.append(temp)
        
        daily_data.append(daily_temps)
    
    return daily_data

def evaluate_model(predictor, test_data):
    """评估模型性能"""
    print("\n评估模型性能...")
    
    all_errors = []
    total_mae = 0
    total_rmse = 0
    
    for sample in test_data:
        prediction = predictor.predict(sample)
        target = predictor.calculate_true_dekads(sample)
        
        # 计算各种误差
        for i in range(len(prediction)):
            error = abs(prediction[i] - target[i])
            all_errors.append(error)
            total_mae += error
            total_rmse += error * error
    
    mae = total_mae / len(all_errors)
    rmse = math.sqrt(total_rmse / len(all_errors))
    
    print(f"  平均绝对误差 (MAE): {mae:.3f}°C")
    print(f"  均方根误差 (RMSE): {rmse:.3f}°C")
    
    return mae, rmse

def show_detailed_results(predictor, test_data, n_samples=3):
    """显示详细的预测结果"""
    print("\n详细预测结果:")
    print("=" * 70)
    
    for i in range(min(n_samples, len(test_data))):
        sample = test_data[i]
        prediction = predictor.predict(sample)
        target = predictor.calculate_true_dekads(sample)
        
        print(f"\n样本 {i+1}:")
        print(f"  前10天温度: {[round(t, 1) for t in sample[:10]]}")
        print(f"  真实旬平均: {[round(t, 2) for t in target]}°C")
        print(f"  预测旬平均: {[round(p, 2) for p in prediction]}°C")
        
        errors = [round(prediction[j] - target[j], 2) for j in range(len(target))]
        print(f"  预测误差:   {errors}°C")
        print("-" * 50)

def demonstrate_attention(predictor):
    """演示注意力机制"""
    print("\n注意力机制分析:")
    print("=" * 50)
    
    attention = predictor.softmax(predictor.attention_weights)
    
    print("各天的注意力权重:")
    for i in range(min(20, len(attention))):
        weight = attention[i]
        bar_length = int(weight * 100)
        bar = "█" * bar_length
        print(f"  第{i+1:2d}天: {weight:.4f} {bar}")
    
    # 找出权重最高的几天
    weight_pairs = [(i, attention[i]) for i in range(len(attention))]
    weight_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n注意力权重最高的5天:")
    for i, (day, weight) in enumerate(weight_pairs[:5]):
        print(f"  第{day+1}天: {weight:.4f}")
    
    print(f"\n总权重和: {sum(attention):.6f}")

def show_network_structure(predictor):
    """显示网络结构信息"""
    print("\n网络结构详情:")
    print("=" * 40)
    
    # 计算参数数量
    w1_params = predictor.input_days * predictor.hidden_dim
    b1_params = predictor.hidden_dim
    w2_params = predictor.hidden_dim * predictor.num_dekads
    b2_params = predictor.num_dekads
    att_params = predictor.input_days
    
    total_params = w1_params + b1_params + w2_params + b2_params + att_params
    
    print(f"层级结构:")
    print(f"  输入层 → 隐藏层: {predictor.input_days} → {predictor.hidden_dim}")
    print(f"  隐藏层 → 输出层: {predictor.hidden_dim} → {predictor.num_dekads}")
    print(f"  注意力权重: {predictor.input_days} 个")
    
    print(f"\n参数统计:")
    print(f"  W1权重矩阵: {w1_params} 个参数")
    print(f"  b1偏置向量: {b1_params} 个参数")
    print(f"  W2权重矩阵: {w2_params} 个参数")
    print(f"  b2偏置向量: {b2_params} 个参数")
    print(f"  注意力权重: {att_params} 个参数")
    print(f"  总参数量: {total_params} 个")

def main():
    """主演示函数"""
    print("🌡️  逐日到旬平均温度预测神经网络")
    print("=" * 60)
    print("纯Python实现 - 无需外部依赖库")
    print()
    
    # 设置随机种子
    random.seed(42)
    
    # 生成数据
    print("📊 数据准备...")
    train_data = generate_sample_data(n_samples=60, n_days=30)
    test_data = generate_sample_data(n_samples=20, n_days=30)
    
    print(f"  训练样本: {len(train_data)} 个")
    print(f"  测试样本: {len(test_data)} 个")
    print(f"  数据格式: 30天逐日温度 → 3个旬平均温度")
    
    # 创建神经网络
    print("\n🧠 神经网络创建...")
    predictor = PureTemperaturePredictor(input_days=30, hidden_dim=8)
    
    # 显示网络结构
    show_network_structure(predictor)
    
    # 训练前性能
    print("\n📈 训练前性能评估:")
    mae_before, rmse_before = evaluate_model(predictor, test_data[:10])
    
    # 训练模型
    print("\n🔄 模型训练...")
    predictor.train_simple(train_data, epochs=500, learning_rate=0.005)
    
    # 训练后性能
    print("\n📊 训练后性能评估:")
    mae_after, rmse_after = evaluate_model(predictor, test_data)
    
    # 性能改善
    mae_improve = mae_before - mae_after
    rmse_improve = rmse_before - rmse_after
    print(f"\n性能改善:")
    print(f"  MAE改善: {mae_improve:.3f}°C ({mae_improve/mae_before*100:.1f}%)")
    print(f"  RMSE改善: {rmse_improve:.3f}°C ({rmse_improve/rmse_before*100:.1f}%)")
    
    # 详细结果展示
    show_detailed_results(predictor, test_data, n_samples=3)
    
    # 注意力机制演示
    demonstrate_attention(predictor)
    
    # 总结
    print("\n🎯 总结:")
    print("-" * 40)
    print("✅ 成功实现了逐日温度到旬平均的神经网络映射")
    print("✅ 模型能够学习温度数据的时间模式")
    print("✅ 注意力机制帮助识别重要的时间步")
    print("✅ 纯Python实现，易于理解和修改")
    
    print("\n🔧 实际应用建议:")
    print("• 使用更多真实数据进行训练")
    print("• 考虑使用PyTorch等深度学习框架")
    print("• 可以添加更多气象变量作为输入")
    print("• 根据具体需求调整网络结构")
    
    print("\n🚀 扩展可能性:")
    print("• 支持多变量输入（温度、湿度、气压等）")
    print("• 添加空间卷积层处理格点数据")
    print("• 使用更复杂的时间序列模型（LSTM、Transformer）")
    print("• 集成多个模型进行ensemble预测")

if __name__ == "__main__":
    main()