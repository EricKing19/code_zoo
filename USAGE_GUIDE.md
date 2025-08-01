# 格点模式数据到站点旬平均温度预测 - 使用指南

## 🎯 项目概述

您的需求是：
- **输入**: GRIB格式的格点模式数据（逐日温度预报）
- **输出**: 站点的自然旬平均温度预测
- **核心挑战**: 格点到站点的空间映射 + 逐日到旬平均的时间聚合

## 📋 已提供的解决方案

### 1. 核心文件
- `gridpoint_to_station_prediction.py` - 主要神经网络模型
- `grib_data_loader.py` - GRIB数据加载工具
- `pure_python_demo.py` - 简化演示版本

### 2. 关键特性
- ✅ **空间注意力机制**: 自动学习格点邻域的重要性
- ✅ **时间卷积网络**: 捕捉逐日温度的时间模式
- ✅ **站点特异性编码**: 为每个站点学习独特的特征
- ✅ **多GRIB格式支持**: pygrib、cfgrib、eccodes

## 🔧 代码修改指南

### 步骤1: 准备您的站点信息

```python
from gridpoint_to_station_prediction import StationInfo

# 修改这部分来匹配您的实际站点
stations = [
    StationInfo(
        station_id="54511",     # 站点编号
        name="北京",            # 站点名称
        lat=39.93,             # 纬度
        lon=116.28,            # 经度
        grid_i=0,              # 格点i坐标（将自动计算）
        grid_j=0,              # 格点j坐标（将自动计算）
        elevation=54.0         # 海拔（可选）
    ),
    # 添加更多站点...
]
```

### 步骤2: 修改GRIB数据加载

在`grib_data_loader.py`中修改`load_real_grib_data`函数：

```python
def load_your_grib_data(grib_file_paths, stations):
    """
    加载您的实际GRIB数据
    """
    from grib_data_loader import GRIBDataLoader
    
    loader = GRIBDataLoader(method='auto')  # 自动选择可用的库
    
    all_grid_data = []
    
    for grib_file in grib_file_paths:
        try:
            # 加载格点数据
            grid_temp = loader.load_grib_file(
                grib_file, 
                parameter='t2m'  # 根据您的GRIB文件调整参数名
            )
            
            # 检查数据维度
            print(f"文件 {grib_file} 数据形状: {grid_temp.shape}")
            
            all_grid_data.append(grid_temp)
            
        except Exception as e:
            print(f"加载文件 {grib_file} 失败: {e}")
            continue
    
    # 合并数据
    if all_grid_data:
        grid_data = np.stack(all_grid_data, axis=0)
        return grid_data
    else:
        raise ValueError("没有成功加载任何GRIB文件")
```

### 步骤3: 数据格式转换

```python
def prepare_training_data(grib_files, stations, days_per_sample=30):
    """
    准备训练数据
    
    Args:
        grib_files: GRIB文件路径列表
        stations: 站点信息列表
        days_per_sample: 每个样本的天数
    """
    
    # 1. 加载GRIB数据
    grid_data = load_your_grib_data(grib_files, stations)
    
    # 2. 数据重组
    # 假设grid_data形状为 (total_days, lat, lon)
    # 需要重组为 (samples, days_per_sample, lat, lon)
    
    total_days = grid_data.shape[0]
    n_samples = total_days // days_per_sample
    
    # 截取完整的样本
    grid_data_reshaped = grid_data[:n_samples * days_per_sample]
    grid_data_reshaped = grid_data_reshaped.reshape(
        n_samples, days_per_sample, grid_data.shape[1], grid_data.shape[2]
    )
    
    # 3. 从格点数据提取站点对应的观测值
    # 这里需要您提供真实的站点观测数据
    station_data = extract_station_observations(
        grid_data_reshaped, stations, days_per_sample
    )
    
    return grid_data_reshaped, station_data

def extract_station_observations(grid_data, stations, days_per_sample):
    """
    提取站点观测数据
    
    这里需要您提供真实的站点观测数据来训练模型
    如果只有格点数据，可以从对应位置提取并添加误差模拟
    """
    n_samples, n_days, lat_size, lon_size = grid_data.shape
    n_stations = len(stations)
    
    station_data = np.zeros((n_samples, n_days, n_stations))
    
    # 方法1: 从最近格点提取（如果没有真实观测）
    for sample_idx in range(n_samples):
        for day_idx in range(n_days):
            for station_idx, station in enumerate(stations):
                # 找到最近格点
                # 这里需要根据您的格点坐标系统调整
                grid_temp = grid_data[sample_idx, day_idx, station.grid_i, station.grid_j]
                
                # 添加站点特异性误差（模拟站点与格点的差异）
                station_bias = np.random.normal(0, 1.0)  # 系统性偏差
                random_error = np.random.normal(0, 0.5)  # 随机误差
                
                station_data[sample_idx, day_idx, station_idx] = grid_temp + station_bias + random_error
    
    # 方法2: 加载真实的站点观测数据（推荐）
    # station_data = load_real_station_observations(stations, dates)
    
    return station_data
```

### 步骤4: 模型训练

```python
from gridpoint_to_station_prediction import GridToStationDataset, GridStationPredictor
from torch.utils.data import DataLoader

def train_your_model(grid_data, station_data, stations):
    """
    训练您的模型
    """
    
    # 1. 创建数据集
    dataset = GridToStationDataset(
        grid_data=grid_data,
        station_temps=station_data,
        stations=stations,
        sequence_length=30,      # 30天输入
        neighbor_radius=2        # 提取5x5邻域
    )
    
    # 2. 数据分割
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 3. 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 4. 创建模型
    predictor = GridStationPredictor(
        input_days=30,
        n_stations=len(stations),
        neighbor_size=(2*2+1)**2,  # 5x5邻域
        hidden_dim=64,
        learning_rate=0.001
    )
    
    # 5. 训练
    print("开始训练...")
    train_losses, val_losses = predictor.train_model(
        train_loader, val_loader, 
        epochs=100, patience=15, verbose=True
    )
    
    # 6. 评估
    station_names = [station.name for station in stations]
    results = predictor.evaluate(test_loader, station_names)
    
    return predictor, results
```

### 步骤5: 实际预测

```python
def predict_dekad_temperatures(model, new_grib_files, stations):
    """
    对新的GRIB数据进行旬平均温度预测
    """
    
    # 1. 加载新的GRIB数据
    new_grid_data = load_your_grib_data(new_grib_files, stations)
    
    # 2. 重组数据格式
    if len(new_grid_data.shape) == 3:
        new_grid_data = new_grid_data[np.newaxis, ...]  # 添加样本维度
    
    # 3. 创建临时数据集（只需格点数据）
    # 这里站点数据可以用dummy数据，因为只做预测
    dummy_station_data = np.zeros((new_grid_data.shape[0], 30, len(stations)))
    
    temp_dataset = GridToStationDataset(
        grid_data=new_grid_data,
        station_temps=dummy_station_data,
        stations=stations,
        sequence_length=30,
        neighbor_radius=2
    )
    
    # 4. 预测
    model.model.eval()
    predictions = []
    
    for i in range(len(temp_dataset)):
        grid_features, _ = temp_dataset[i]
        grid_features = grid_features.unsqueeze(0)  # 添加batch维度
        
        with torch.no_grad():
            pred = model.model(grid_features.to(model.device))
            predictions.append(pred.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    
    # 5. 返回结果
    return predictions  # 形状: (samples, dekads, stations)
```

## 🗂️ 完整使用流程

### 1. 环境准备

```bash
# 安装基础依赖
pip install torch numpy pandas matplotlib scikit-learn

# 安装GRIB读取库（选择一个）
pip install pygrib          # 推荐
# 或者
pip install cfgrib xarray   # 替代方案
# 或者
pip install eccodes-python  # 另一个选择
```

### 2. 数据准备

```python
# 准备您的数据
grib_files = [
    "/path/to/your/grib_day1.grib",
    "/path/to/your/grib_day2.grib",
    # ... 更多文件
]

stations = [
    # 您的站点信息
]

# 加载和准备数据
grid_data, station_data = prepare_training_data(grib_files, stations)
```

### 3. 模型训练

```python
# 训练模型
model, results = train_your_model(grid_data, station_data, stations)

# 保存模型
torch.save(model.model.state_dict(), 'your_grid_station_model.pth')
```

### 4. 预测使用

```python
# 加载训练好的模型
model.model.load_state_dict(torch.load('your_grid_station_model.pth'))

# 对新数据进行预测
new_grib_files = ["/path/to/new/data.grib"]
dekad_predictions = predict_dekad_temperatures(model, new_grib_files, stations)

print(f"预测结果形状: {dekad_predictions.shape}")
# 输出: (samples, 3_dekads, n_stations)
```

## ⚠️ 重要注意事项

### 1. 数据质量
- **站点观测数据**: 需要真实的站点观测数据来训练模型
- **格点坐标**: 确保正确匹配站点位置与格点坐标
- **数据单位**: 检查温度单位（Kelvin vs Celsius）

### 2. 模型参数调整
```python
# 根据您的数据调整这些参数
neighbor_radius = 2      # 格点邻域大小
hidden_dim = 64         # 隐藏层维度
learning_rate = 0.001   # 学习率
batch_size = 16         # 批量大小
epochs = 100           # 训练轮数
```

### 3. 坐标系统匹配
```python
def find_nearest_grid_point(station_lat, station_lon, grid_lats, grid_lons):
    """
    找到站点最近的格点
    需要根据您的具体格点坐标系统调整
    """
    lat_diff = np.abs(grid_lats - station_lat)
    lon_diff = np.abs(grid_lons - station_lon)
    
    lat_idx = np.argmin(lat_diff)
    lon_idx = np.argmin(lon_diff)
    
    return lat_idx, lon_idx
```

### 4. 自然旬定义
```python
def calculate_natural_dekads(daily_temps):
    """
    计算自然旬平均
    
    自然旬定义：
    - 上旬：1-10日
    - 中旬：11-20日  
    - 下旬：21日-月末
    """
    # 当前实现是简单的10天分组
    # 您可以根据实际需求修改为真正的自然旬
    pass
```

## 🚀 快速开始示例

```python
# 完整的快速开始示例
if __name__ == "__main__":
    # 1. 定义站点
    stations = [
        StationInfo("54511", "北京", 39.93, 116.28, 0, 0, 54.0),
        # 添加更多站点...
    ]
    
    # 2. GRIB文件路径
    grib_files = ["your_file1.grib", "your_file2.grib"]
    
    # 3. 训练模型
    try:
        grid_data, station_data = prepare_training_data(grib_files, stations)
        model, results = train_your_model(grid_data, station_data, stations)
        print("训练完成！")
        
        # 4. 预测新数据
        new_predictions = predict_dekad_temperatures(model, ["new_data.grib"], stations)
        print(f"预测完成，结果形状: {new_predictions.shape}")
        
    except Exception as e:
        print(f"运行失败: {e}")
        print("请检查数据路径和格式")
```

这个解决方案专门针对您的需求进行了设计，包含了从GRIB格点数据到站点旬平均温度预测的完整流程。您只需要根据实际数据格式调整相应的加载和预处理代码即可。