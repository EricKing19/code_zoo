import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class DailyToDekadDataset(Dataset):
    """
    数据集类：将逐日温度数据转换为旬平均温度的训练数据
    """
    def __init__(self, daily_temps: np.ndarray, sequence_length: int = 30):
        """
        Args:
            daily_temps: 逐日温度数据 (samples, days)
            sequence_length: 输入序列长度（天数）
        """
        self.daily_temps = daily_temps
        self.sequence_length = sequence_length
        
        # 计算旬平均温度（每10天为一旬）
        self.dekad_temps = self._calculate_dekad_averages(daily_temps)
        
    def _calculate_dekad_averages(self, daily_temps: np.ndarray) -> np.ndarray:
        """计算旬平均温度"""
        n_samples, n_days = daily_temps.shape
        n_dekads = n_days // 10  # 每10天为一旬
        
        dekad_temps = []
        for sample in daily_temps:
            sample_dekads = []
            for i in range(n_dekads):
                start_day = i * 10
                end_day = min((i + 1) * 10, n_days)
                dekad_avg = np.mean(sample[start_day:end_day])
                sample_dekads.append(dekad_avg)
            dekad_temps.append(sample_dekads)
        
        return np.array(dekad_temps)
    
    def __len__(self):
        return len(self.daily_temps)
    
    def __getitem__(self, idx):
        daily_temp = torch.FloatTensor(self.daily_temps[idx])
        dekad_temp = torch.FloatTensor(self.dekad_temps[idx])
        return daily_temp, dekad_temp

class TemporalConvNet(nn.Module):
    """
    时间卷积网络：用于处理逐日温度序列
    """
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

class DailyToDekadPredictor(nn.Module):
    """
    主要模型：将逐日温度预测转换为旬平均温度
    """
    def __init__(self, input_days: int, num_dekads: int, hidden_dim: int = 64, dropout: float = 0.1):
        super(DailyToDekadPredictor, self).__init__()
        
        self.input_days = input_days
        self.num_dekads = num_dekads
        
        # 时间卷积层用于特征提取
        self.temporal_conv = TemporalConvNet(
            input_channels=1,
            num_channels=[hidden_dim, hidden_dim, hidden_dim//2],
            kernel_size=3,
            dropout=dropout
        )
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, input_days),
            nn.Softmax(dim=1)
        )
        
        # 输出层
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim//2 + input_days, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_dekads)
        )
        
    def forward(self, daily_temps):
        batch_size = daily_temps.size(0)
        
        # 添加通道维度 (batch, 1, days)
        x = daily_temps.unsqueeze(1)
        
        # 时间卷积特征提取
        conv_features = self.temporal_conv(x)  # (batch, hidden_dim//2, days)
        
        # 全局特征
        global_features = self.global_avg_pool(conv_features).squeeze(-1)  # (batch, hidden_dim//2)
        
        # 注意力权重
        attention_weights = self.attention(global_features)  # (batch, input_days)
        
        # 加权的日温度特征
        weighted_daily = torch.sum(daily_temps * attention_weights, dim=1, keepdim=True)  # (batch, 1)
        weighted_daily = weighted_daily.expand(-1, daily_temps.size(1))  # (batch, input_days)
        
        # 特征融合
        combined_features = torch.cat([global_features, weighted_daily], dim=1)
        
        # 输出旬平均温度
        dekad_predictions = self.output_layers(combined_features)
        
        return dekad_predictions

class TemperaturePredictor:
    """
    温度预测器：包含训练、评估和推理功能
    """
    def __init__(self, input_days: int = 30, hidden_dim: int = 64, learning_rate: float = 0.001):
        self.input_days = input_days
        self.num_dekads = input_days // 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.model = DailyToDekadPredictor(
            input_days=input_days,
            num_dekads=self.num_dekads,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # 优化器和损失函数
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, 
                   epochs: int = 100, patience: int = 10, verbose: bool = True):
        """
        训练模型
        """
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            for daily_temps, dekad_temps in train_loader:
                daily_temps = daily_temps.to(self.device)
                dekad_temps = dekad_temps.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(daily_temps)
                loss = self.criterion(predictions, dekad_temps)
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
                for daily_temps, dekad_temps in val_loader:
                    daily_temps = daily_temps.to(self.device)
                    dekad_temps = dekad_temps.to(self.device)
                    
                    predictions = self.model(daily_temps)
                    loss = self.criterion(predictions, dekad_temps)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # 学习率调度
            self.scheduler.step()
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        return train_losses, val_losses
    
    def evaluate(self, test_loader: DataLoader):
        """
        评估模型性能
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for daily_temps, dekad_temps in test_loader:
                daily_temps = daily_temps.to(self.device)
                predictions = self.model(daily_temps)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(dekad_temps.numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # 计算评估指标
        mse = mean_squared_error(all_targets.flatten(), all_predictions.flatten())
        mae = mean_absolute_error(all_targets.flatten(), all_predictions.flatten())
        r2 = r2_score(all_targets.flatten(), all_predictions.flatten())
        rmse = np.sqrt(mse)
        
        print(f"评估结果:")
        print(f"RMSE: {rmse:.4f}°C")
        print(f"MAE: {mae:.4f}°C")
        print(f"R²: {r2:.4f}")
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2, 'predictions': all_predictions, 'targets': all_targets}
    
    def predict(self, daily_temps: np.ndarray) -> np.ndarray:
        """
        对新数据进行预测
        """
        self.model.eval()
        
        if len(daily_temps.shape) == 1:
            daily_temps = daily_temps.reshape(1, -1)
        
        daily_temps = torch.FloatTensor(daily_temps).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(daily_temps)
        
        return predictions.cpu().numpy()

def generate_synthetic_data(n_samples: int = 1000, n_days: int = 30, 
                          temp_range: Tuple[float, float] = (15.0, 35.0),
                          seasonal_amplitude: float = 10.0) -> np.ndarray:
    """
    生成合成的逐日温度数据用于演示
    """
    np.random.seed(42)
    
    daily_temps = []
    for i in range(n_samples):
        # 基础温度趋势
        base_temp = np.random.uniform(temp_range[0], temp_range[1])
        
        # 季节性变化
        day_of_year = np.random.randint(1, 366)
        seasonal_pattern = seasonal_amplitude * np.sin(2 * np.pi * (day_of_year + np.arange(n_days)) / 365)
        
        # 添加随机噪声和短期变化
        noise = np.random.normal(0, 2.0, n_days)
        short_term_variation = np.random.normal(0, 1.0, n_days)
        
        # 组合所有因素
        sample_temps = base_temp + seasonal_pattern + noise + short_term_variation
        daily_temps.append(sample_temps)
    
    return np.array(daily_temps)

def plot_results(predictions: np.ndarray, targets: np.ndarray, n_samples: int = 5):
    """
    可视化预测结果
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 显示几个样本的预测结果
    for i in range(min(n_samples, len(predictions))):
        ax = axes[i//3, i%3] if i < 6 else None
        if ax is not None:
            dekad_days = np.arange(1, len(predictions[i]) + 1)
            ax.plot(dekad_days, targets[i], 'o-', label='真实值', linewidth=2)
            ax.plot(dekad_days, predictions[i], 's-', label='预测值', linewidth=2)
            ax.set_title(f'样本 {i+1}')
            ax.set_xlabel('旬数')
            ax.set_ylabel('温度 (°C)')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # 删除多余的子图
    if n_samples < 6:
        for i in range(n_samples, 6):
            axes[i//3, i%3].remove()
    
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    主函数：演示模型的使用
    """
    print("正在生成合成数据...")
    # 生成合成数据
    daily_data = generate_synthetic_data(n_samples=1000, n_days=30)
    
    # 创建数据集
    dataset = DailyToDekadDataset(daily_data, sequence_length=30)
    
    # 数据分割
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("正在初始化模型...")
    # 初始化预测器
    predictor = TemperaturePredictor(input_days=30, hidden_dim=64, learning_rate=0.001)
    
    print("开始训练模型...")
    # 训练模型
    train_losses, val_losses = predictor.train_model(
        train_loader, val_loader, epochs=100, patience=15, verbose=True
    )
    
    print("正在评估模型...")
    # 评估模型
    results = predictor.evaluate(test_loader)
    
    print("正在生成可视化结果...")
    # 可视化结果
    plot_results(results['predictions'], results['targets'])
    
    print("训练完成！模型已保存为 'best_model.pth'")
    print("预测结果图表已保存为 'prediction_results.png'")

if __name__ == "__main__":
    main()