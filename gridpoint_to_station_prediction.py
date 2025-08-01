"""
格点模式数据到站点旬平均温度预测神经网络
输入：GRIB格式的格点逐日温度数据
输出：站点的自然旬平均温度
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class StationInfo:
    """站点信息类"""
    def __init__(self, station_id: str, name: str, lat: float, lon: float, 
                 grid_i: int, grid_j: int, elevation: float = 0.0):
        self.station_id = station_id
        self.name = name
        self.lat = lat
        self.lon = lon
        self.grid_i = grid_i  # 最近格点的i坐标
        self.grid_j = grid_j  # 最近格点的j坐标
        self.elevation = elevation

class GridToStationDataset(Dataset):
    """
    格点到站点的数据集类
    """
    def __init__(self, grid_data: np.ndarray, station_temps: np.ndarray, 
                 stations: List[StationInfo], sequence_length: int = 30,
                 neighbor_radius: int = 2):
        """
        Args:
            grid_data: 格点数据 (samples, days, lat_grid, lon_grid)
            station_temps: 站点温度数据 (samples, days, n_stations)
            stations: 站点信息列表
            sequence_length: 输入序列长度（天数）
            neighbor_radius: 提取格点邻域的半径
        """
        self.grid_data = grid_data
        self.station_temps = station_temps
        self.stations = stations
        self.sequence_length = sequence_length
        self.neighbor_radius = neighbor_radius
        
        # 为每个站点提取格点邻域数据
        self.station_grid_features = self._extract_station_features()
        
        # 计算站点旬平均温度
        self.station_dekad_temps = self._calculate_station_dekads()
        
    def _extract_station_features(self) -> np.ndarray:
        """为每个站点提取周围格点的特征"""
        n_samples, n_days, lat_size, lon_size = self.grid_data.shape
        n_stations = len(self.stations)
        neighbor_size = (2 * self.neighbor_radius + 1) ** 2
        
        # 存储每个站点的格点邻域特征
        station_features = np.zeros((n_samples, n_days, n_stations, neighbor_size))
        
        for sample_idx in range(n_samples):
            for day_idx in range(n_days):
                for station_idx, station in enumerate(self.stations):
                    # 提取站点周围的格点邻域
                    i_start = max(0, station.grid_i - self.neighbor_radius)
                    i_end = min(lat_size, station.grid_i + self.neighbor_radius + 1)
                    j_start = max(0, station.grid_j - self.neighbor_radius)
                    j_end = min(lon_size, station.grid_j + self.neighbor_radius + 1)
                    
                    # 提取邻域数据并展平
                    neighborhood = self.grid_data[sample_idx, day_idx, i_start:i_end, j_start:j_end]
                    
                    # 如果邻域不足，用最近格点值填充
                    if neighborhood.size < neighbor_size:
                        padded = np.full(neighbor_size, self.grid_data[sample_idx, day_idx, station.grid_i, station.grid_j])
                        padded[:neighborhood.size] = neighborhood.flatten()
                        station_features[sample_idx, day_idx, station_idx] = padded
                    else:
                        station_features[sample_idx, day_idx, station_idx] = neighborhood.flatten()[:neighbor_size]
        
        return station_features
    
    def _calculate_station_dekads(self) -> np.ndarray:
        """计算站点旬平均温度"""
        n_samples, n_days, n_stations = self.station_temps.shape
        n_dekads = n_days // 10
        
        dekad_temps = np.zeros((n_samples, n_dekads, n_stations))
        
        for sample_idx in range(n_samples):
            for station_idx in range(n_stations):
                for dekad_idx in range(n_dekads):
                    start_day = dekad_idx * 10
                    end_day = min((dekad_idx + 1) * 10, n_days)
                    dekad_avg = np.mean(self.station_temps[sample_idx, start_day:end_day, station_idx])
                    dekad_temps[sample_idx, dekad_idx, station_idx] = dekad_avg
        
        return dekad_temps
    
    def __len__(self):
        return len(self.grid_data)
    
    def __getitem__(self, idx):
        grid_features = torch.FloatTensor(self.station_grid_features[idx])  # (days, n_stations, neighbor_features)
        station_dekads = torch.FloatTensor(self.station_dekad_temps[idx])  # (n_dekads, n_stations)
        return grid_features, station_dekads

class SpatialAttention(nn.Module):
    """空间注意力机制：学习格点邻域的重要性权重"""
    def __init__(self, neighbor_size: int, hidden_dim: int = 32):
        super(SpatialAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(neighbor_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, neighbor_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        # x: (batch, days, stations, neighbor_features)
        attention_weights = self.attention(x)
        weighted_features = x * attention_weights
        return weighted_features.sum(dim=-1)  # (batch, days, stations)

class TemporalConvNet(nn.Module):
    """时间卷积网络"""
    def __init__(self, input_channels: int, num_channels: list, kernel_size: int = 3, dropout: float = 0.1):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, 
                         stride=1, dilation=dilation_size, 
                         padding=(kernel_size-1) * dilation_size // 2)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class GridToStationPredictor(nn.Module):
    """
    格点到站点的温度预测模型
    """
    def __init__(self, input_days: int, n_stations: int, neighbor_size: int, 
                 hidden_dim: int = 64, dropout: float = 0.1):
        super(GridToStationPredictor, self).__init__()
        
        self.input_days = input_days
        self.n_stations = n_stations
        self.num_dekads = input_days // 10
        
        # 空间注意力：处理格点邻域
        self.spatial_attention = SpatialAttention(neighbor_size, hidden_dim//2)
        
        # 时间卷积：处理时间序列
        self.temporal_conv = TemporalConvNet(
            input_channels=n_stations,
            num_channels=[hidden_dim, hidden_dim, hidden_dim//2],
            kernel_size=3,
            dropout=dropout
        )
        
        # 全局池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 时间注意力：学习不同时间步的重要性
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, input_days),
            nn.Softmax(dim=1)
        )
        
        # 站点特异性编码
        self.station_embedding = nn.Parameter(torch.randn(n_stations, hidden_dim//4))
        
        # 输出层：为每个站点预测旬平均
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim//2 + hidden_dim//4 + input_days, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, self.num_dekads)
            ) for _ in range(n_stations)
        ])
        
    def forward(self, grid_features):
        batch_size = grid_features.size(0)
        
        # 空间注意力：聚合格点邻域特征
        spatial_features = self.spatial_attention(grid_features)  # (batch, days, stations)
        
        # 时间卷积：提取时间特征
        temporal_input = spatial_features.transpose(1, 2)  # (batch, stations, days)
        conv_features = self.temporal_conv(temporal_input)  # (batch, hidden_dim//2, days)
        
        # 全局特征
        global_features = self.global_avg_pool(conv_features).squeeze(-1)  # (batch, hidden_dim//2)
        
        # 时间注意力
        temporal_attn = self.temporal_attention(global_features)  # (batch, input_days)
        
        # 加权时间特征
        weighted_temporal = torch.sum(spatial_features * temporal_attn.unsqueeze(-1), dim=1)  # (batch, stations)
        
        # 为每个站点单独预测
        station_predictions = []
        for station_idx in range(self.n_stations):
            # 站点特异性特征
            station_emb = self.station_embedding[station_idx].expand(batch_size, -1)
            
            # 站点特定的输入特征
            station_temporal = weighted_temporal[:, station_idx:station_idx+1].expand(-1, self.input_days)
            
            # 特征融合
            station_input = torch.cat([global_features, station_emb, station_temporal], dim=1)
            
            # 站点预测
            station_pred = self.output_layers[station_idx](station_input)
            station_predictions.append(station_pred)
        
        # 堆叠所有站点的预测结果
        predictions = torch.stack(station_predictions, dim=2)  # (batch, dekads, stations)
        
        return predictions

class GridStationPredictor:
    """
    格点到站点的温度预测器
    """
    def __init__(self, input_days: int = 30, n_stations: int = 10, neighbor_size: int = 9,
                 hidden_dim: int = 64, learning_rate: float = 0.001):
        self.input_days = input_days
        self.n_stations = n_stations
        self.neighbor_size = neighbor_size
        self.num_dekads = input_days // 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.model = GridToStationPredictor(
            input_days=input_days,
            n_stations=n_stations,
            neighbor_size=neighbor_size,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # 优化器和损失函数
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, 
                   epochs: int = 100, patience: int = 15, verbose: bool = True):
        """训练模型"""
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            for grid_features, station_dekads in train_loader:
                grid_features = grid_features.to(self.device)
                station_dekads = station_dekads.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(grid_features)
                loss = self.criterion(predictions, station_dekads)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for grid_features, station_dekads in val_loader:
                    grid_features = grid_features.to(self.device)
                    station_dekads = station_dekads.to(self.device)
                    
                    predictions = self.model(grid_features)
                    loss = self.criterion(predictions, station_dekads)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # 学习率调度
            self.scheduler.step()
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_grid_station_model.pth')
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_grid_station_model.pth'))
        
        return train_losses, val_losses
    
    def evaluate(self, test_loader: DataLoader, station_names: List[str] = None):
        """评估模型性能"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for grid_features, station_dekads in test_loader:
                grid_features = grid_features.to(self.device)
                predictions = self.model(grid_features)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(station_dekads.numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # 整体评估指标
        mse = mean_squared_error(all_targets.flatten(), all_predictions.flatten())
        mae = mean_absolute_error(all_targets.flatten(), all_predictions.flatten())
        r2 = r2_score(all_targets.flatten(), all_predictions.flatten())
        rmse = np.sqrt(mse)
        
        print(f"整体评估结果:")
        print(f"RMSE: {rmse:.4f}°C")
        print(f"MAE: {mae:.4f}°C")
        print(f"R²: {r2:.4f}")
        
        # 按站点评估
        print(f"\n各站点评估结果:")
        station_metrics = {}
        for station_idx in range(self.n_stations):
            station_pred = all_predictions[:, :, station_idx].flatten()
            station_target = all_targets[:, :, station_idx].flatten()
            
            station_mae = mean_absolute_error(station_target, station_pred)
            station_rmse = np.sqrt(mean_squared_error(station_target, station_pred))
            station_r2 = r2_score(station_target, station_pred)
            
            station_name = station_names[station_idx] if station_names else f"Station_{station_idx+1}"
            station_metrics[station_name] = {
                'mae': station_mae, 'rmse': station_rmse, 'r2': station_r2
            }
            
            print(f"  {station_name}: MAE={station_mae:.3f}°C, RMSE={station_rmse:.3f}°C, R²={station_r2:.3f}")
        
        return {
            'overall': {'rmse': rmse, 'mae': mae, 'r2': r2},
            'stations': station_metrics,
            'predictions': all_predictions, 
            'targets': all_targets
        }
    
    def predict(self, grid_features: np.ndarray) -> np.ndarray:
        """对新的格点数据进行预测"""
        self.model.eval()
        
        if len(grid_features.shape) == 3:
            grid_features = grid_features.reshape(1, *grid_features.shape)
        
        grid_features = torch.FloatTensor(grid_features).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(grid_features)
        
        return predictions.cpu().numpy()

def load_grib_data(grib_file_paths: List[str], stations: List[StationInfo]) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载GRIB格点数据的占位函数
    实际使用时需要根据具体的GRIB格式进行修改
    """
    print("注意：这是一个示例函数，实际使用时需要根据您的GRIB数据格式进行修改")
    
    # 这里是示例数据生成，实际使用时请替换为真实的GRIB数据加载
    # 可以使用 pygrib, xarray, eccodes 等库来读取GRIB文件
    
    n_samples = len(grib_file_paths)
    n_days = 30
    lat_size, lon_size = 50, 60  # 根据实际格点大小调整
    n_stations = len(stations)
    
    # 生成示例格点数据
    grid_data = np.random.randn(n_samples, n_days, lat_size, lon_size) * 5 + 25
    
    # 生成示例站点数据（从对应格点位置提取并添加噪声）
    station_data = np.zeros((n_samples, n_days, n_stations))
    for sample_idx in range(n_samples):
        for day_idx in range(n_days):
            for station_idx, station in enumerate(stations):
                # 从最近格点提取温度并添加站点特异性误差
                grid_temp = grid_data[sample_idx, day_idx, station.grid_i, station.grid_j]
                station_bias = np.random.normal(0, 1)  # 站点系统性偏差
                random_error = np.random.normal(0, 0.5)  # 随机误差
                station_data[sample_idx, day_idx, station_idx] = grid_temp + station_bias + random_error
    
    return grid_data, station_data

def create_sample_stations() -> List[StationInfo]:
    """创建示例站点信息"""
    stations = [
        StationInfo("54511", "北京", 39.93, 116.28, 25, 30, 54.0),
        StationInfo("58457", "上海", 31.17, 121.43, 20, 35, 7.0),
        StationInfo("59287", "广州", 23.13, 113.32, 15, 28, 41.0),
        StationInfo("56294", "成都", 30.67, 104.02, 18, 25, 506.0),
        StationInfo("53698", "西安", 34.30, 108.93, 22, 26, 397.0),
        StationInfo("54342", "济南", 36.68, 117.00, 24, 31, 170.0),
        StationInfo("57494", "武汉", 30.62, 114.13, 19, 29, 23.0),
        StationInfo("56778", "重庆", 29.52, 106.48, 17, 24, 259.0),
        StationInfo("53614", "兰州", 36.05, 103.88, 23, 23, 1517.0),
        StationInfo("56985", "长沙", 28.23, 112.93, 16, 28, 68.0),
    ]
    return stations

def generate_sample_data(n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray, List[StationInfo]]:
    """生成示例数据"""
    print(f"生成 {n_samples} 个样本的格点和站点数据...")
    
    stations = create_sample_stations()
    
    # 模拟GRIB文件路径
    grib_files = [f"sample_data_{i:03d}.grib" for i in range(n_samples)]
    
    # 加载数据
    grid_data, station_data = load_grib_data(grib_files, stations)
    
    return grid_data, station_data, stations

def visualize_station_results(predictions: np.ndarray, targets: np.ndarray, 
                            stations: List[StationInfo], n_samples: int = 3):
    """可视化站点预测结果"""
    n_stations = len(stations)
    n_dekads = predictions.shape[1]
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(15, 5*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx in range(min(n_samples, len(predictions))):
        # 左图：所有站点的旬平均对比
        ax1 = axes[sample_idx, 0]
        dekad_indices = np.arange(1, n_dekads + 1)
        
        for station_idx in range(min(5, n_stations)):  # 只显示前5个站点
            station_name = stations[station_idx].name
            ax1.plot(dekad_indices, targets[sample_idx, :, station_idx], 
                    'o-', label=f'{station_name} 真实', linewidth=2, markersize=6)
            ax1.plot(dekad_indices, predictions[sample_idx, :, station_idx], 
                    's--', label=f'{station_name} 预测', linewidth=2, markersize=6)
        
        ax1.set_title(f'样本 {sample_idx+1} - 站点旬平均温度预测')
        ax1.set_xlabel('旬数')
        ax1.set_ylabel('温度 (°C)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：误差分析
        ax2 = axes[sample_idx, 1]
        errors = predictions[sample_idx] - targets[sample_idx]
        station_names = [s.name for s in stations[:min(5, n_stations)]]
        
        for dekad_idx in range(n_dekads):
            ax2.bar([name + f'_旬{dekad_idx+1}' for name in station_names], 
                   errors[dekad_idx, :min(5, n_stations)], 
                   alpha=0.7, label=f'第{dekad_idx+1}旬')
        
        ax2.set_title(f'样本 {sample_idx+1} - 预测误差分析')
        ax2.set_xlabel('站点_旬')
        ax2.set_ylabel('预测误差 (°C)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('station_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数：演示格点到站点的预测"""
    print("格点模式数据到站点旬平均温度预测")
    print("=" * 60)
    
    # 生成示例数据
    grid_data, station_data, stations = generate_sample_data(n_samples=150)
    
    print(f"数据信息:")
    print(f"  格点数据形状: {grid_data.shape}")
    print(f"  站点数据形状: {station_data.shape}")
    print(f"  站点数量: {len(stations)}")
    
    # 创建数据集
    dataset = GridToStationDataset(
        grid_data=grid_data, 
        station_temps=station_data, 
        stations=stations,
        sequence_length=30,
        neighbor_radius=2
    )
    
    print(f"  每个站点格点邻域大小: {(2*2+1)**2} 个格点")
    
    # 数据分割
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"\n数据分割:")
    print(f"  训练集: {train_size} 个样本")
    print(f"  验证集: {val_size} 个样本")
    print(f"  测试集: {test_size} 个样本")
    
    # 创建预测器
    predictor = GridStationPredictor(
        input_days=30, 
        n_stations=len(stations), 
        neighbor_size=(2*2+1)**2,
        hidden_dim=64, 
        learning_rate=0.001
    )
    
    print(f"\n模型结构:")
    print(f"  输入: 30天 × {len(stations)}个站点 × {(2*2+1)**2}个邻域格点")
    print(f"  输出: 3个旬 × {len(stations)}个站点")
    
    # 训练模型
    print("\n开始训练...")
    train_losses, val_losses = predictor.train_model(
        train_loader, val_loader, epochs=50, patience=10, verbose=True
    )
    
    # 评估模型
    print("\n模型评估:")
    station_names = [station.name for station in stations]
    results = predictor.evaluate(test_loader, station_names)
    
    # 可视化结果
    print("\n生成可视化结果...")
    visualize_station_results(
        results['predictions'][:3], 
        results['targets'][:3], 
        stations, 
        n_samples=3
    )
    
    print("\n训练完成！")
    print("模型已保存为 'best_grid_station_model.pth'")
    print("预测结果图表已保存为 'station_prediction_results.png'")

if __name__ == "__main__":
    main()