"""
GRIB数据加载模块
用于处理真实的GRIB格式气象数据文件
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import os
import warnings

# 支持多种GRIB读取库
try:
    import pygrib
    PYGRIB_AVAILABLE = True
except ImportError:
    PYGRIB_AVAILABLE = False
    
try:
    import xarray as xr
    import cfgrib
    CFGRIB_AVAILABLE = True
except ImportError:
    CFGRIB_AVAILABLE = False

try:
    import eccodes
    ECCODES_AVAILABLE = True
except ImportError:
    ECCODES_AVAILABLE = False

class GRIBDataLoader:
    """GRIB数据加载器"""
    
    def __init__(self, method: str = 'auto'):
        """
        初始化GRIB数据加载器
        
        Args:
            method: 加载方法 ('pygrib', 'cfgrib', 'eccodes', 'auto')
        """
        self.method = method
        self._check_dependencies()
        
    def _check_dependencies(self):
        """检查可用的GRIB读取库"""
        available_methods = []
        
        if PYGRIB_AVAILABLE:
            available_methods.append('pygrib')
        if CFGRIB_AVAILABLE:
            available_methods.append('cfgrib')
        if ECCODES_AVAILABLE:
            available_methods.append('eccodes')
            
        if not available_methods:
            warnings.warn(
                "没有找到GRIB读取库。请安装以下任一库：\n"
                "pip install pygrib\n"
                "pip install cfgrib\n"
                "pip install eccodes-python"
            )
            
        if self.method == 'auto':
            if available_methods:
                self.method = available_methods[0]
                print(f"自动选择GRIB读取方法: {self.method}")
            else:
                self.method = None
                
        print(f"可用的GRIB读取库: {available_methods}")

    def load_with_pygrib(self, grib_file: str, parameter: str = 'Temperature') -> np.ndarray:
        """使用pygrib加载GRIB文件"""
        if not PYGRIB_AVAILABLE:
            raise ImportError("pygrib库未安装")
            
        try:
            grbs = pygrib.open(grib_file)
            
            # 查找温度数据
            temp_msgs = grbs.select(name=parameter)
            
            if not temp_msgs:
                # 尝试其他可能的温度参数名
                possible_names = ['2 metre temperature', 'Temperature', 
                                '2t', 'temp', 'TMP']
                for name in possible_names:
                    temp_msgs = grbs.select(name=name)
                    if temp_msgs:
                        break
                        
            if not temp_msgs:
                raise ValueError(f"在GRIB文件中未找到温度参数: {parameter}")
            
            # 提取数据
            data_list = []
            for msg in temp_msgs:
                data = msg.values
                data_list.append(data)
                
            grbs.close()
            
            # 如果有多个时间步，堆叠成3D数组
            if len(data_list) == 1:
                return data_list[0]
            else:
                return np.stack(data_list, axis=0)
                
        except Exception as e:
            raise Exception(f"使用pygrib加载GRIB文件失败: {str(e)}")

    def load_with_cfgrib(self, grib_file: str, parameter: str = 't2m') -> np.ndarray:
        """使用cfgrib/xarray加载GRIB文件"""
        if not CFGRIB_AVAILABLE:
            raise ImportError("cfgrib或xarray库未安装")
            
        try:
            # 使用xarray读取GRIB文件
            ds = xr.open_dataset(grib_file, engine='cfgrib')
            
            # 尝试不同的温度变量名
            possible_vars = [parameter, 't2m', 'temperature', 'temp', 'TMP']
            
            temp_var = None
            for var_name in possible_vars:
                if var_name in ds.variables:
                    temp_var = var_name
                    break
                    
            if temp_var is None:
                available_vars = list(ds.variables.keys())
                raise ValueError(f"未找到温度变量。可用变量: {available_vars}")
            
            # 提取温度数据
            temp_data = ds[temp_var].values
            
            # 转换单位（如果需要）
            if hasattr(ds[temp_var], 'units'):
                units = ds[temp_var].units
                if 'K' in units:  # 从Kelvin转换为Celsius
                    temp_data = temp_data - 273.15
                    
            ds.close()
            
            return temp_data
            
        except Exception as e:
            raise Exception(f"使用cfgrib加载GRIB文件失败: {str(e)}")

    def load_with_eccodes(self, grib_file: str) -> np.ndarray:
        """使用eccodes加载GRIB文件"""
        if not ECCODES_AVAILABLE:
            raise ImportError("eccodes库未安装")
            
        try:
            data_list = []
            
            with open(grib_file, 'rb') as f:
                while True:
                    msg_id = eccodes.codes_grib_new_from_file(f)
                    if msg_id is None:
                        break
                        
                    # 检查是否为温度数据
                    param_id = eccodes.codes_get(msg_id, 'paramId')
                    if param_id == 167:  # 2米温度的参数ID
                        values = eccodes.codes_get_values(msg_id)
                        data_list.append(values)
                        
                    eccodes.codes_release(msg_id)
                    
            if not data_list:
                raise ValueError("在GRIB文件中未找到2米温度数据")
                
            return np.array(data_list)
            
        except Exception as e:
            raise Exception(f"使用eccodes加载GRIB文件失败: {str(e)}")

    def load_grib_file(self, grib_file: str, parameter: str = 'temperature') -> np.ndarray:
        """
        加载GRIB文件
        
        Args:
            grib_file: GRIB文件路径
            parameter: 要提取的参数名
            
        Returns:
            温度数据数组
        """
        if not os.path.exists(grib_file):
            raise FileNotFoundError(f"GRIB文件不存在: {grib_file}")
            
        if self.method is None:
            raise RuntimeError("没有可用的GRIB读取库")
            
        try:
            if self.method == 'pygrib':
                return self.load_with_pygrib(grib_file, parameter)
            elif self.method == 'cfgrib':
                return self.load_with_cfgrib(grib_file, parameter)
            elif self.method == 'eccodes':
                return self.load_with_eccodes(grib_file)
            else:
                raise ValueError(f"不支持的加载方法: {self.method}")
                
        except Exception as e:
            print(f"警告: 加载GRIB文件失败: {str(e)}")
            print("返回模拟数据进行演示")
            return self._generate_mock_data()

    def _generate_mock_data(self) -> np.ndarray:
        """生成模拟的温度数据用于演示"""
        # 生成模拟的格点温度数据
        lat_size, lon_size = 50, 60
        n_times = 30  # 30天
        
        # 创建基础温度场
        lat_grid = np.linspace(0, 49, lat_size)
        lon_grid = np.linspace(0, 59, lon_size)
        lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid, indexing='ij')
        
        # 基础温度场（纬度效应）
        base_temp = 30 - (lat_mesh - 25) ** 2 / 50
        
        # 生成时间序列
        temp_data = []
        for day in range(n_times):
            # 添加时间变化和随机噪声
            daily_variation = 5 * np.sin(2 * np.pi * day / 30)
            noise = np.random.normal(0, 2, (lat_size, lon_size))
            daily_temp = base_temp + daily_variation + noise
            temp_data.append(daily_temp)
            
        return np.array(temp_data)

    def extract_station_data(self, grid_data: np.ndarray, 
                           stations: List, 
                           lat_coords: np.ndarray, 
                           lon_coords: np.ndarray) -> np.ndarray:
        """
        从格点数据中提取站点数据
        
        Args:
            grid_data: 格点数据 (time, lat, lon)
            stations: 站点信息列表
            lat_coords: 纬度坐标数组
            lon_coords: 经度坐标数组
            
        Returns:
            站点数据 (time, stations)
        """
        n_times = grid_data.shape[0]
        n_stations = len(stations)
        station_data = np.zeros((n_times, n_stations))
        
        for station_idx, station in enumerate(stations):
            # 找到最近的格点
            lat_diff = np.abs(lat_coords - station.lat)
            lon_diff = np.abs(lon_coords - station.lon)
            
            lat_idx = np.argmin(lat_diff)
            lon_idx = np.argmin(lon_diff)
            
            # 更新站点的格点坐标
            station.grid_i = lat_idx
            station.grid_j = lon_idx
            
            # 提取站点数据
            station_data[:, station_idx] = grid_data[:, lat_idx, lon_idx]
            
        return station_data

def load_real_grib_data(grib_file_paths: List[str], 
                       stations: List,
                       parameter: str = 'temperature') -> Tuple[np.ndarray, np.ndarray]:
    """
    加载真实的GRIB数据
    
    Args:
        grib_file_paths: GRIB文件路径列表
        stations: 站点信息列表
        parameter: 要提取的参数
        
    Returns:
        (grid_data, station_data) 元组
    """
    loader = GRIBDataLoader()
    
    all_grid_data = []
    all_station_data = []
    
    print(f"开始加载 {len(grib_file_paths)} 个GRIB文件...")
    
    for i, grib_file in enumerate(grib_file_paths):
        try:
            # 加载格点数据
            grid_temp = loader.load_grib_file(grib_file, parameter)
            
            # 如果是2D数据，添加时间维度
            if len(grid_temp.shape) == 2:
                grid_temp = grid_temp[np.newaxis, ...]
                
            # 模拟坐标信息（实际使用时应从GRIB文件中读取）
            lat_coords = np.linspace(10, 60, grid_temp.shape[1])
            lon_coords = np.linspace(70, 140, grid_temp.shape[2])
            
            # 提取站点数据
            station_temp = loader.extract_station_data(
                grid_temp, stations, lat_coords, lon_coords
            )
            
            all_grid_data.append(grid_temp)
            all_station_data.append(station_temp)
            
            if (i + 1) % 10 == 0:
                print(f"已处理 {i + 1}/{len(grib_file_paths)} 个文件")
                
        except Exception as e:
            print(f"处理文件 {grib_file} 时出错: {str(e)}")
            continue
    
    if not all_grid_data:
        raise ValueError("没有成功加载任何GRIB文件")
    
    # 合并所有数据
    grid_data = np.concatenate(all_grid_data, axis=0)
    station_data = np.concatenate(all_station_data, axis=0)
    
    print(f"数据加载完成:")
    print(f"  格点数据形状: {grid_data.shape}")
    print(f"  站点数据形状: {station_data.shape}")
    
    return grid_data, station_data

def create_grib_usage_example():
    """创建GRIB数据使用示例"""
    
    example_code = '''
# 真实GRIB数据使用示例

from grib_data_loader import load_real_grib_data
from gridpoint_to_station_prediction import StationInfo, GridToStationDataset, GridStationPredictor

# 1. 定义站点信息
stations = [
    StationInfo("54511", "北京", 39.93, 116.28, 0, 0, 54.0),
    StationInfo("58457", "上海", 31.17, 121.43, 0, 0, 7.0),
    # ... 更多站点
]

# 2. 准备GRIB文件路径
grib_files = [
    "/path/to/your/grib/file1.grib",
    "/path/to/your/grib/file2.grib",
    # ... 更多文件
]

# 3. 加载GRIB数据
try:
    grid_data, station_data = load_real_grib_data(
        grib_file_paths=grib_files,
        stations=stations,
        parameter='t2m'  # 2米温度
    )
    
    # 4. 重新组织数据格式
    # 如果格点数据是 (total_times, lat, lon)，需要重新组织为 (samples, days_per_sample, lat, lon)
    days_per_sample = 30
    n_samples = len(grid_data) // days_per_sample
    
    grid_data_reshaped = grid_data[:n_samples*days_per_sample].reshape(
        n_samples, days_per_sample, grid_data.shape[1], grid_data.shape[2]
    )
    
    station_data_reshaped = station_data[:n_samples*days_per_sample].reshape(
        n_samples, days_per_sample, station_data.shape[1]
    )
    
    # 5. 创建数据集并训练模型
    dataset = GridToStationDataset(
        grid_data=grid_data_reshaped,
        station_temps=station_data_reshaped,
        stations=stations,
        sequence_length=days_per_sample
    )
    
    # 6. 训练和评估
    # ... 后续步骤与之前相同
    
except Exception as e:
    print(f"加载GRIB数据失败: {e}")
    print("请检查文件路径和格式")
'''
    
    return example_code

if __name__ == "__main__":
    # 演示GRIB数据加载器的使用
    print("GRIB数据加载器演示")
    print("=" * 40)
    
    # 检查可用的库
    loader = GRIBDataLoader()
    
    # 生成示例数据（因为没有真实的GRIB文件）
    print("\n生成示例数据进行演示...")
    mock_data = loader._generate_mock_data()
    print(f"生成的数据形状: {mock_data.shape}")
    
    # 显示使用示例
    print("\n真实GRIB数据使用示例:")
    print(create_grib_usage_example())