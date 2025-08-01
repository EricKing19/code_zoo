"""
使用示例：如何使用训练好的神经网络进行逐日到旬平均温度预测
"""

import numpy as np
import torch
from daily_to_dekad_temperature_prediction import TemperaturePredictor, DailyToDekadDataset
import matplotlib.pyplot as plt

def load_your_daily_temperature_data():
    """
    替换这个函数来加载您的实际逐日温度数据
    
    返回值应该是一个numpy数组，形状为 (samples, days)
    例如：如果您有100个格点，每个格点30天的数据，返回形状应该是 (100, 30)
    """
    # 这里是示例数据生成，请替换为您的实际数据加载代码
    n_samples = 50  # 例如50个格点
    n_days = 30     # 30天的数据
    
    # 示例：从文件加载数据
    # daily_temps = np.load('your_daily_temperature_data.npy')
    
    # 或者从CSV文件加载
    # import pandas as pd
    # df = pd.read_csv('your_temperature_data.csv')
    # daily_temps = df.values
    
    # 示例数据（请替换为实际数据）
    np.random.seed(123)
    daily_temps = []
    for i in range(n_samples):
        base_temp = np.random.uniform(20, 30)
        trend = np.linspace(-2, 2, n_days)
        noise = np.random.normal(0, 1, n_days)
        temps = base_temp + trend + noise
        daily_temps.append(temps)
    
    return np.array(daily_temps)

def predict_dekad_temperatures():
    """
    使用训练好的模型预测旬平均温度
    """
    print("1. 加载逐日温度数据...")
    daily_temps = load_your_daily_temperature_data()
    print(f"   数据形状: {daily_temps.shape}")
    print(f"   温度范围: {daily_temps.min():.2f}°C 到 {daily_temps.max():.2f}°C")
    
    print("\n2. 初始化预测器...")
    predictor = TemperaturePredictor(input_days=30, hidden_dim=64)
    
    print("\n3. 加载训练好的模型...")
    try:
        predictor.model.load_state_dict(torch.load('best_model.pth', map_location=predictor.device))
        print("   模型加载成功!")
    except FileNotFoundError:
        print("   未找到训练好的模型文件 'best_model.pth'")
        print("   请先运行主训练脚本: python daily_to_dekad_temperature_prediction.py")
        return
    
    print("\n4. 进行旬平均温度预测...")
    dekad_predictions = predictor.predict(daily_temps)
    print(f"   预测结果形状: {dekad_predictions.shape}")
    print(f"   旬平均温度范围: {dekad_predictions.min():.2f}°C 到 {dekad_predictions.max():.2f}°C")
    
    # 计算真实的旬平均温度用于比较
    n_dekads = daily_temps.shape[1] // 10
    true_dekad_temps = []
    for sample in daily_temps:
        sample_dekads = []
        for i in range(n_dekads):
            start_day = i * 10
            end_day = min((i + 1) * 10, daily_temps.shape[1])
            dekad_avg = np.mean(sample[start_day:end_day])
            sample_dekads.append(dekad_avg)
        true_dekad_temps.append(sample_dekads)
    true_dekad_temps = np.array(true_dekad_temps)
    
    print("\n5. 可视化预测结果...")
    visualize_predictions(daily_temps, true_dekad_temps, dekad_predictions)
    
    return dekad_predictions

def visualize_predictions(daily_temps, true_dekad_temps, predicted_dekad_temps, n_samples=6):
    """
    可视化预测结果
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i in range(min(n_samples, len(daily_temps))):
        ax = axes[i]
        
        # 绘制逐日温度
        days = np.arange(1, len(daily_temps[i]) + 1)
        ax.plot(days, daily_temps[i], 'lightblue', alpha=0.7, linewidth=1, label='逐日温度')
        
        # 绘制真实旬平均温度
        dekad_centers = np.arange(5, len(daily_temps[i]), 10)[:len(true_dekad_temps[i])]
        ax.plot(dekad_centers, true_dekad_temps[i], 'o-', color='blue', 
                linewidth=3, markersize=8, label='真实旬平均')
        
        # 绘制预测旬平均温度
        ax.plot(dekad_centers, predicted_dekad_temps[i], 's-', color='red', 
                linewidth=3, markersize=8, label='预测旬平均')
        
        ax.set_title(f'格点 {i+1} - 温度预测结果', fontsize=12)
        ax.set_xlabel('天数', fontsize=10)
        ax.set_ylabel('温度 (°C)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 添加误差信息
        mae = np.mean(np.abs(true_dekad_temps[i] - predicted_dekad_temps[i]))
        ax.text(0.02, 0.98, f'MAE: {mae:.2f}°C', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('usage_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 整体误差统计
    overall_mae = np.mean(np.abs(true_dekad_temps - predicted_dekad_temps))
    overall_rmse = np.sqrt(np.mean((true_dekad_temps - predicted_dekad_temps)**2))
    
    print(f"\n整体预测误差:")
    print(f"  平均绝对误差 (MAE): {overall_mae:.3f}°C")
    print(f"  均方根误差 (RMSE): {overall_rmse:.3f}°C")

def batch_prediction_example():
    """
    批量预测示例：处理大量格点数据
    """
    print("\n=== 批量预测示例 ===")
    
    # 模拟大量格点数据
    n_gridpoints = 1000  # 1000个格点
    n_days = 30
    
    print(f"生成 {n_gridpoints} 个格点的逐日温度数据...")
    
    # 生成示例数据（替换为您的实际数据加载）
    np.random.seed(456)
    large_daily_data = []
    for i in range(n_gridpoints):
        base_temp = np.random.uniform(15, 35)
        seasonal_trend = 5 * np.sin(2 * np.pi * np.arange(n_days) / 365)
        noise = np.random.normal(0, 2, n_days)
        temps = base_temp + seasonal_trend + noise
        large_daily_data.append(temps)
    
    large_daily_data = np.array(large_daily_data)
    
    # 初始化预测器
    predictor = TemperaturePredictor(input_days=30, hidden_dim=64)
    
    try:
        predictor.model.load_state_dict(torch.load('best_model.pth', map_location=predictor.device))
    except FileNotFoundError:
        print("请先运行主训练脚本训练模型")
        return
    
    print("开始批量预测...")
    
    # 分批处理以节省内存
    batch_size = 100
    all_predictions = []
    
    for i in range(0, n_gridpoints, batch_size):
        end_idx = min(i + batch_size, n_gridpoints)
        batch_data = large_daily_data[i:end_idx]
        batch_predictions = predictor.predict(batch_data)
        all_predictions.append(batch_predictions)
        
        if (i // batch_size + 1) % 5 == 0:
            print(f"  已处理 {end_idx}/{n_gridpoints} 个格点")
    
    # 合并所有预测结果
    all_predictions = np.vstack(all_predictions)
    
    print(f"批量预测完成！")
    print(f"  输入数据形状: {large_daily_data.shape}")
    print(f"  预测结果形状: {all_predictions.shape}")
    print(f"  预测温度范围: {all_predictions.min():.2f}°C 到 {all_predictions.max():.2f}°C")
    
    # 保存结果
    np.save('batch_dekad_predictions.npy', all_predictions)
    print("  预测结果已保存为 'batch_dekad_predictions.npy'")

def save_predictions_to_file():
    """
    将预测结果保存到不同格式的文件
    """
    print("\n=== 保存预测结果示例 ===")
    
    # 加载数据和进行预测
    daily_temps = load_your_daily_temperature_data()
    
    predictor = TemperaturePredictor(input_days=30, hidden_dim=64)
    try:
        predictor.model.load_state_dict(torch.load('best_model.pth', map_location=predictor.device))
    except FileNotFoundError:
        print("请先训练模型")
        return
    
    dekad_predictions = predictor.predict(daily_temps)
    
    # 保存为numpy格式
    np.save('dekad_temperature_predictions.npy', dekad_predictions)
    print("预测结果已保存为 'dekad_temperature_predictions.npy'")
    
    # 保存为CSV格式
    import pandas as pd
    
    # 创建DataFrame
    dekad_columns = [f'旬{i+1}' for i in range(dekad_predictions.shape[1])]
    df = pd.DataFrame(dekad_predictions, columns=dekad_columns)
    df.index.name = '格点编号'
    
    df.to_csv('dekad_temperature_predictions.csv')
    print("预测结果已保存为 'dekad_temperature_predictions.csv'")
    
    # 显示前几行
    print("\n预测结果示例:")
    print(df.head())

if __name__ == "__main__":
    print("逐日到旬平均温度预测 - 使用示例")
    print("=" * 50)
    
    # 单次预测示例
    dekad_temps = predict_dekad_temperatures()
    
    # 批量预测示例
    batch_prediction_example()
    
    # 保存结果示例
    save_predictions_to_file()
    
    print("\n所有示例运行完成！")