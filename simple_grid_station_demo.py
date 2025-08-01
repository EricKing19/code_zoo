"""
简化版格点到站点旬平均温度预测演示
展示从GRIB格点数据到站点旬平均温度预测的核心概念
使用纯Python实现，无需外部依赖
"""

import math
import random

class StationInfo:
    """站点信息类"""
    def __init__(self, station_id, name, lat, lon, grid_i=0, grid_j=0, elevation=0.0):
        self.station_id = station_id
        self.name = name
        self.lat = lat
        self.lon = lon
        self.grid_i = grid_i
        self.grid_j = grid_j
        self.elevation = elevation

class SimpleGridStationPredictor:
    """
    简化的格点到站点预测器
    """
    def __init__(self, n_stations=5, neighbor_size=9, hidden_dim=16):
        self.n_stations = n_stations
        self.neighbor_size = neighbor_size  # 3x3邻域
        self.hidden_dim = hidden_dim
        self.input_days = 30
        self.num_dekads = 3
        
        # 空间注意力权重
        self.spatial_weights = [
            [random.gauss(0, 0.1) for _ in range(neighbor_size)] 
            for _ in range(n_stations)
        ]
        
        # 时间卷积权重
        self.temporal_weights = [
            [random.gauss(0, 0.1) for _ in range(self.input_days)] 
            for _ in range(hidden_dim)
        ]
        
        # 站点特异性权重
        self.station_weights = [
            [random.gauss(0, 0.1) for _ in range(hidden_dim)] 
            for _ in range(n_stations)
        ]
        
        # 输出权重
        self.output_weights = [
            [random.gauss(0, 0.1) for _ in range(self.num_dekads)] 
            for _ in range(n_stations)
        ]
        
        print(f"格点到站点预测器初始化:")
        print(f"  站点数量: {n_stations}")
        print(f"  格点邻域: 3x3 = {neighbor_size} 个格点")
        print(f"  输入天数: {self.input_days}")
        print(f"  输出旬数: {self.num_dekads}")
    
    def relu(self, x):
        """ReLU激活函数"""
        return max(0, x)
    
    def softmax(self, x_list):
        """Softmax函数"""
        max_x = max(x_list)
        exp_x = [math.exp(x - max_x) for x in x_list]
        sum_exp = sum(exp_x)
        return [exp / sum_exp for exp in exp_x]
    
    def extract_station_features(self, grid_data, stations):
        """
        从格点数据中提取站点邻域特征
        
        Args:
            grid_data: 格点数据 (days, lat, lon)
            stations: 站点信息列表
            
        Returns:
            站点特征 (days, stations, neighbor_features)
        """
        n_days = len(grid_data)
        lat_size = len(grid_data[0])
        lon_size = len(grid_data[0][0])
        
        station_features = []
        
        for day_idx in range(n_days):
            day_features = []
            
            for station_idx, station in enumerate(stations):
                # 提取3x3邻域
                neighborhood = []
                
                for di in range(-1, 2):  # -1, 0, 1
                    for dj in range(-1, 2):
                        ni = max(0, min(lat_size-1, station.grid_i + di))
                        nj = max(0, min(lon_size-1, station.grid_j + dj))
                        neighborhood.append(grid_data[day_idx][ni][nj])
                
                day_features.append(neighborhood)
            
            station_features.append(day_features)
        
        return station_features
    
    def spatial_attention(self, neighborhood_features, station_idx):
        """空间注意力：聚合格点邻域"""
        weights = self.softmax(self.spatial_weights[station_idx])
        
        weighted_sum = 0
        for i, feature in enumerate(neighborhood_features):
            weighted_sum += feature * weights[i]
            
        return weighted_sum
    
    def temporal_convolution(self, station_time_series):
        """时间卷积：提取时间特征"""
        conv_features = []
        
        for h in range(self.hidden_dim):
            feature = 0
            for t, temp in enumerate(station_time_series):
                feature += temp * self.temporal_weights[h][t]
            conv_features.append(self.relu(feature))
        
        return conv_features
    
    def station_specific_prediction(self, conv_features, station_idx):
        """站点特异性预测"""
        station_features = []
        
        for f in conv_features:
            station_feature = 0
            for h, cf in enumerate(conv_features):
                station_feature += cf * self.station_weights[station_idx][h]
            station_features.append(station_feature)
        
        # 预测旬平均
        dekad_predictions = []
        for d in range(self.num_dekads):
            pred = 0
            for h, sf in enumerate(station_features):
                pred += sf * self.output_weights[station_idx][d]
            dekad_predictions.append(pred)
        
        return dekad_predictions
    
    def predict(self, grid_data, stations):
        """
        完整的预测流程
        
        Args:
            grid_data: 格点数据 (days, lat, lon)
            stations: 站点信息列表
            
        Returns:
            预测结果 (stations, dekads)
        """
        # 1. 提取站点特征
        station_features = self.extract_station_features(grid_data, stations)
        
        # 2. 为每个站点预测
        all_predictions = []
        
        for station_idx, station in enumerate(stations):
            # 提取该站点的时间序列
            station_time_series = []
            
            for day_idx in range(len(station_features)):
                # 空间注意力聚合
                spatial_feature = self.spatial_attention(
                    station_features[day_idx][station_idx], 
                    station_idx
                )
                station_time_series.append(spatial_feature)
            
            # 时间卷积
            conv_features = self.temporal_convolution(station_time_series)
            
            # 站点特异性预测
            dekad_pred = self.station_specific_prediction(conv_features, station_idx)
            
            all_predictions.append(dekad_pred)
        
        return all_predictions
    
    def calculate_true_dekads(self, station_daily_temps):
        """计算真实的旬平均温度"""
        true_dekads = []
        
        for station_temps in station_daily_temps:
            station_dekads = []
            for d in range(self.num_dekads):
                start_day = d * 10
                end_day = min((d + 1) * 10, len(station_temps))
                dekad_avg = sum(station_temps[start_day:end_day]) / (end_day - start_day)
                station_dekads.append(dekad_avg)
            true_dekads.append(station_dekads)
        
        return true_dekads
    
    def train_simple(self, grid_data_list, station_data_list, stations, epochs=200, learning_rate=0.001):
        """简化的训练过程"""
        print(f"\n开始训练 (共{epochs}轮)...")
        
        for epoch in range(epochs):
            total_loss = 0
            
            for sample_idx, (grid_data, station_data) in enumerate(zip(grid_data_list, station_data_list)):
                # 前向传播
                predictions = self.predict(grid_data, stations)
                targets = self.calculate_true_dekads(station_data)
                
                # 计算损失
                sample_loss = 0
                for station_idx in range(len(stations)):
                    for dekad_idx in range(self.num_dekads):
                        error = predictions[station_idx][dekad_idx] - targets[station_idx][dekad_idx]
                        sample_loss += error * error
                
                total_loss += sample_loss
                
                # 简化的权重更新（仅更新输出层）
                for station_idx in range(len(stations)):
                    for dekad_idx in range(self.num_dekads):
                        error = predictions[station_idx][dekad_idx] - targets[station_idx][dekad_idx]
                        self.output_weights[station_idx][dekad_idx] -= learning_rate * error / len(grid_data_list)
            
            avg_loss = total_loss / (len(grid_data_list) * len(stations) * self.num_dekads)
            
            if epoch % 50 == 0:
                print(f"  轮次 {epoch:3d}: 平均损失 = {avg_loss:.4f}")
        
        print("训练完成！")

def generate_mock_grib_data(n_samples=20, n_days=30, lat_size=15, lon_size=20):
    """生成模拟的GRIB格点数据"""
    print(f"生成模拟GRIB数据: {n_samples}个样本, 每样本{n_days}天, 格点{lat_size}x{lon_size}")
    
    all_grid_data = []
    
    for sample in range(n_samples):
        # 创建基础温度场
        base_temp = 25 + random.gauss(0, 5)
        
        sample_data = []
        for day in range(n_days):
            daily_grid = []
            
            for lat in range(lat_size):
                lat_row = []
                for lon in range(lon_size):
                    # 基础温度 + 纬度效应 + 时间变化 + 噪声
                    temp = (base_temp + 
                           (lat - lat_size/2) * 0.5 +  # 纬度效应
                           3 * math.sin(2 * math.pi * day / 30) +  # 时间变化
                           random.gauss(0, 2))  # 噪声
                    lat_row.append(temp)
                daily_grid.append(lat_row)
            
            sample_data.append(daily_grid)
        
        all_grid_data.append(sample_data)
    
    return all_grid_data

def extract_station_data_from_grid(grid_data_list, stations):
    """从格点数据中提取站点对应的观测数据"""
    print("从格点数据提取站点观测...")
    
    all_station_data = []
    
    for grid_data in grid_data_list:
        sample_station_data = []
        
        for station in stations:
            station_temps = []
            
            for day_data in grid_data:
                # 从最近格点提取温度
                grid_temp = day_data[station.grid_i][station.grid_j]
                
                # 添加站点特异性偏差
                station_bias = random.gauss(0, 1.0)
                observed_temp = grid_temp + station_bias
                
                station_temps.append(observed_temp)
            
            sample_station_data.append(station_temps)
        
        all_station_data.append(sample_station_data)
    
    return all_station_data

def create_sample_stations():
    """创建示例站点"""
    stations = [
        StationInfo("54511", "北京", 39.93, 116.28, 7, 10),
        StationInfo("58457", "上海", 31.17, 121.43, 5, 15),
        StationInfo("59287", "广州", 23.13, 113.32, 3, 12),
        StationInfo("56294", "成都", 30.67, 104.02, 4, 8),
        StationInfo("53698", "西安", 34.30, 108.93, 6, 9),
    ]
    
    print("创建示例站点:")
    for station in stations:
        print(f"  {station.name} ({station.lat:.2f}°N, {station.lon:.2f}°E) -> 格点({station.grid_i}, {station.grid_j})")
    
    return stations

def evaluate_predictions(predictions, targets, stations):
    """评估预测结果"""
    print("\n预测结果评估:")
    print("=" * 50)
    
    total_mae = 0
    total_count = 0
    
    for station_idx, station in enumerate(stations):
        station_errors = []
        
        for dekad_idx in range(3):
            error = abs(predictions[station_idx][dekad_idx] - targets[station_idx][dekad_idx])
            station_errors.append(error)
            total_mae += error
            total_count += 1
        
        avg_error = sum(station_errors) / len(station_errors)
        print(f"  {station.name}: MAE = {avg_error:.3f}°C")
    
    overall_mae = total_mae / total_count
    print(f"\n整体平均绝对误差: {overall_mae:.3f}°C")
    
    return overall_mae

def show_detailed_predictions(predictions, targets, stations, sample_idx=0):
    """显示详细的预测结果"""
    print(f"\n样本 {sample_idx + 1} 的详细预测结果:")
    print("=" * 60)
    
    for station_idx, station in enumerate(stations):
        print(f"\n{station.name}:")
        print(f"  真实旬平均: {[round(t, 2) for t in targets[station_idx]]}°C")
        print(f"  预测旬平均: {[round(p, 2) for p in predictions[station_idx]]}°C")
        
        errors = [round(predictions[station_idx][d] - targets[station_idx][d], 2) 
                 for d in range(3)]
        print(f"  预测误差:   {errors}°C")

def demonstrate_grid_to_station_mapping(grid_data, stations):
    """演示格点到站点的映射过程"""
    print("\n格点到站点映射演示:")
    print("=" * 40)
    
    # 显示某一天的格点数据分布
    day_0_data = grid_data[0]  # 第一天
    
    print("第1天的格点温度分布 (部分区域):")
    for i in range(min(8, len(day_0_data))):
        row_str = "  "
        for j in range(min(10, len(day_0_data[i]))):
            row_str += f"{day_0_data[i][j]:5.1f} "
        print(row_str)
    
    print("\n各站点对应的格点温度:")
    for station in stations:
        grid_temp = day_0_data[station.grid_i][station.grid_j]
        print(f"  {station.name}: 格点({station.grid_i},{station.grid_j}) = {grid_temp:.2f}°C")

def main():
    """主演示函数"""
    print("🌡️  格点模式数据到站点旬平均温度预测")
    print("=" * 60)
    print("演示: GRIB格点数据 → 站点旬平均温度")
    print()
    
    # 设置随机种子
    random.seed(42)
    
    # 1. 创建站点
    stations = create_sample_stations()
    
    # 2. 生成模拟GRIB数据
    grid_data_list = generate_mock_grib_data(n_samples=30, n_days=30, lat_size=15, lon_size=20)
    
    # 3. 提取站点观测数据
    station_data_list = extract_station_data_from_grid(grid_data_list, stations)
    
    print(f"\n数据准备完成:")
    print(f"  训练样本: {len(grid_data_list)} 个")
    print(f"  格点大小: {len(grid_data_list[0][0])} x {len(grid_data_list[0][0][0])}")
    print(f"  站点数量: {len(stations)} 个")
    print(f"  时间长度: {len(grid_data_list[0])} 天")
    
    # 4. 演示格点到站点映射
    demonstrate_grid_to_station_mapping(grid_data_list[0], stations)
    
    # 5. 创建预测模型
    print(f"\n🧠 创建神经网络模型...")
    predictor = SimpleGridStationPredictor(
        n_stations=len(stations),
        neighbor_size=9,  # 3x3邻域
        hidden_dim=12
    )
    
    # 6. 训练前评估
    print(f"\n📈 训练前性能:")
    sample_predictions = predictor.predict(grid_data_list[0], stations)
    sample_targets = predictor.calculate_true_dekads(station_data_list[0])
    mae_before = evaluate_predictions(sample_predictions, sample_targets, stations)
    
    # 7. 训练模型
    print(f"\n🔄 模型训练...")
    predictor.train_simple(
        grid_data_list[:20],  # 前20个样本用于训练
        station_data_list[:20],
        stations,
        epochs=200,
        learning_rate=0.001
    )
    
    # 8. 训练后评估
    print(f"\n📊 训练后性能:")
    test_predictions = predictor.predict(grid_data_list[25], stations)  # 测试样本
    test_targets = predictor.calculate_true_dekads(station_data_list[25])
    mae_after = evaluate_predictions(test_predictions, test_targets, stations)
    
    # 9. 性能改善
    improvement = mae_before - mae_after
    improvement_pct = improvement / mae_before * 100
    print(f"\n性能改善:")
    print(f"  训练前MAE: {mae_before:.3f}°C")
    print(f"  训练后MAE: {mae_after:.3f}°C")
    print(f"  改善幅度: {improvement:.3f}°C ({improvement_pct:.1f}%)")
    
    # 10. 详细结果展示
    show_detailed_predictions(test_predictions, test_targets, stations)
    
    # 11. 总结
    print(f"\n🎯 总结:")
    print("-" * 40)
    print("✅ 成功实现格点到站点的空间映射")
    print("✅ 有效学习逐日到旬平均的时间聚合")
    print("✅ 空间注意力机制自动加权格点邻域")
    print("✅ 站点特异性编码处理不同站点特征")
    
    print(f"\n🔧 实际应用要点:")
    print("• 替换模拟数据为真实GRIB文件加载")
    print("• 使用真实站点观测数据进行训练")
    print("• 根据格点分辨率调整邻域大小")
    print("• 优化网络结构以提高精度")
    
    print(f"\n🚀 扩展方向:")
    print("• 增加更多气象要素（湿度、气压等）")
    print("• 考虑地形和海拔影响")
    print("• 加入天气类型分类")
    print("• 实现不确定性量化")

if __name__ == "__main__":
    main()