# -*- coding: utf-8 -*-
"""
hetao_ag.core.config - 配置管理模块
=====================================

提供灵活的配置管理功能，支持YAML/JSON配置文件和环境变量覆盖。

主要功能:
    - ConfigManager: 配置管理器，支持分层配置和动态更新
    - 支持YAML和JSON格式的配置文件
    - 环境变量优先覆盖（便于部署时注入配置）
    - 配置验证和默认值
    - 嵌套键访问（如 "database.host"）

设计原则:
    - 配置与代码分离，提高可复现性
    - 支持多环境配置（开发、测试、生产）
    - 敏感信息通过环境变量注入

作者: Hetao College
版本: 1.0.0
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from copy import deepcopy
import warnings

# 尝试导入YAML支持
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    warnings.warn("PyYAML未安装，YAML配置文件支持不可用。使用 pip install pyyaml 安装。")


class ConfigError(Exception):
    """配置错误异常
    
    当配置加载、解析或验证失败时抛出。
    """
    pass


class ConfigManager:
    """配置管理器
    
    提供分层配置管理，支持从文件加载配置，并允许环境变量覆盖。
    
    特性:
        - 支持YAML和JSON配置文件
        - 支持嵌套键访问（点分隔符）
        - 环境变量自动覆盖
        - 配置验证和默认值
        - 配置合并和继承
    
    环境变量覆盖规则:
        - 配置键转换为大写
        - 点分隔符转换为双下划线
        - 例如: "database.host" -> "DATABASE__HOST"
    
    属性:
        config: 配置字典
        config_file: 配置文件路径
        
    示例:
        >>> # 从YAML文件加载
        >>> config = ConfigManager("config.yaml")
        >>> db_host = config.get("database.host", "localhost")
        
        >>> # 环境变量覆盖
        >>> os.environ["DATABASE__HOST"] = "production-server"
        >>> db_host = config.get("database.host")  # 返回 "production-server"
        
        >>> # 设置和更新配置
        >>> config.set("api.timeout", 30)
    """
    
    def __init__(
        self,
        config_file: Optional[Union[str, Path]] = None,
        defaults: Optional[Dict[str, Any]] = None,
        env_prefix: str = "",
        auto_reload: bool = False
    ):
        """初始化配置管理器
        
        参数:
            config_file: 配置文件路径（YAML或JSON）
            defaults: 默认配置字典
            env_prefix: 环境变量前缀（用于区分不同应用）
            auto_reload: 是否自动重新加载配置文件（文件变化时）
            
        异常:
            ConfigError: 配置文件加载失败
        """
        self._config: Dict[str, Any] = {}
        self._defaults: Dict[str, Any] = defaults or {}
        self._env_prefix = env_prefix.upper() + "__" if env_prefix else ""
        self._config_file: Optional[Path] = None
        self._auto_reload = auto_reload
        self._last_modified: Optional[float] = None
        
        # 先应用默认配置
        if defaults:
            self._config = deepcopy(defaults)
        
        # 加载配置文件
        if config_file:
            self.load(config_file)
    
    def load(self, config_file: Union[str, Path]) -> 'ConfigManager':
        """加载配置文件
        
        支持YAML(.yaml, .yml)和JSON(.json)格式。
        
        参数:
            config_file: 配置文件路径
            
        返回:
            self，支持链式调用
            
        异常:
            ConfigError: 文件不存在或格式不支持
        """
        path = Path(config_file)
        
        if not path.exists():
            raise ConfigError(f"配置文件不存在: {path}")
        
        self._config_file = path
        self._last_modified = path.stat().st_mtime
        
        suffix = path.suffix.lower()
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if suffix in ('.yaml', '.yml'):
                    if not YAML_AVAILABLE:
                        raise ConfigError("需要安装PyYAML才能加载YAML文件")
                    file_config = yaml.safe_load(f) or {}
                elif suffix == '.json':
                    file_config = json.load(f)
                else:
                    raise ConfigError(f"不支持的配置文件格式: {suffix}")
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigError(f"配置文件解析失败: {e}")
        
        # 合并配置（文件配置覆盖默认值）
        self._config = self._deep_merge(self._config, file_config)
        
        return self
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """深度合并两个字典
        
        参数:
            base: 基础字典
            override: 覆盖字典
            
        返回:
            合并后的新字典
        """
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        支持点分隔的嵌套键访问。环境变量优先于配置文件。
        
        参数:
            key: 配置键（支持点分隔，如 "database.host"）
            default: 默认值（如果配置不存在）
            
        返回:
            配置值
            
        示例:
            >>> config.get("database.host", "localhost")
            >>> config.get("logging.level", "INFO")
        """
        # 检查是否需要重新加载
        if self._auto_reload:
            self._check_reload()
        
        # 首先检查环境变量
        env_key = self._to_env_key(key)
        env_value = os.getenv(env_key)
        if env_value is not None:
            return self._parse_env_value(env_value)
        
        # 然后从配置字典获取
        value = self._get_nested(self._config, key)
        
        if value is not None:
            return value
        
        # 最后检查默认值字典
        default_value = self._get_nested(self._defaults, key)
        if default_value is not None:
            return default_value
        
        return default
    
    def _to_env_key(self, key: str) -> str:
        """将配置键转换为环境变量名
        
        参数:
            key: 配置键
            
        返回:
            环境变量名
        """
        # 点分隔符转换为双下划线，全部大写
        env_key = key.replace(".", "__").upper()
        return self._env_prefix + env_key
    
    def _parse_env_value(self, value: str) -> Any:
        """解析环境变量值
        
        尝试将字符串解析为适当的类型（bool, int, float）。
        
        参数:
            value: 环境变量字符串值
            
        返回:
            解析后的值
        """
        # 布尔值
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        if value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # 尝试解析为数字
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # JSON数组或对象
        if value.startswith(('[', '{')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        return value
    
    def _get_nested(self, data: Dict, key: str) -> Any:
        """获取嵌套字典中的值
        
        参数:
            data: 字典
            key: 点分隔的键
            
        返回:
            值或None
        """
        parts = key.split(".")
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    
    def set(self, key: str, value: Any) -> 'ConfigManager':
        """设置配置值
        
        参数:
            key: 配置键（支持点分隔）
            value: 配置值
            
        返回:
            self，支持链式调用
            
        示例:
            >>> config.set("api.timeout", 30).set("api.retries", 3)
        """
        parts = key.split(".")
        current = self._config
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
        return self
    
    def _check_reload(self):
        """检查并重新加载配置文件（如果已修改）"""
        if self._config_file and self._config_file.exists():
            current_mtime = self._config_file.stat().st_mtime
            if current_mtime != self._last_modified:
                # 保存当前设置的值
                runtime_config = deepcopy(self._config)
                # 重新加载
                self.load(self._config_file)
                # 合并运行时设置（运行时设置优先）
                self._config = self._deep_merge(self._config, runtime_config)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """获取配置段落
        
        参数:
            section: 段落名称
            
        返回:
            段落配置字典的副本
            
        示例:
            >>> db_config = config.get_section("database")
            >>> print(db_config)  # {'host': 'localhost', 'port': 5432}
        """
        value = self.get(section, {})
        if isinstance(value, dict):
            return deepcopy(value)
        return {}
    
    def has(self, key: str) -> bool:
        """检查配置键是否存在
        
        参数:
            key: 配置键
            
        返回:
            是否存在
        """
        return self.get(key) is not None
    
    def all(self) -> Dict[str, Any]:
        """获取所有配置的副本
        
        返回:
            配置字典的深拷贝
        """
        return deepcopy(self._config)
    
    def save(self, path: Optional[Union[str, Path]] = None):
        """保存配置到文件
        
        参数:
            path: 保存路径（默认为原配置文件路径）
            
        异常:
            ConfigError: 无法确定保存路径或写入失败
        """
        save_path = Path(path) if path else self._config_file
        
        if not save_path:
            raise ConfigError("未指定保存路径")
        
        suffix = save_path.suffix.lower()
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                if suffix in ('.yaml', '.yml'):
                    if not YAML_AVAILABLE:
                        raise ConfigError("需要安装PyYAML才能保存YAML文件")
                    yaml.dump(self._config, f, allow_unicode=True, default_flow_style=False)
                elif suffix == '.json':
                    json.dump(self._config, f, ensure_ascii=False, indent=2)
                else:
                    raise ConfigError(f"不支持的配置文件格式: {suffix}")
        except IOError as e:
            raise ConfigError(f"保存配置失败: {e}")
    
    def validate(self, required_keys: List[str]) -> bool:
        """验证必需的配置键是否存在
        
        参数:
            required_keys: 必需的配置键列表
            
        返回:
            是否全部存在
            
        异常:
            ConfigError: 缺少必需的配置
        """
        missing = [key for key in required_keys if not self.has(key)]
        
        if missing:
            raise ConfigError(f"缺少必需的配置: {', '.join(missing)}")
        
        return True
    
    def __getitem__(self, key: str) -> Any:
        """字典风格访问"""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value
    
    def __setitem__(self, key: str, value: Any):
        """字典风格设置"""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """支持 'in' 操作符"""
        return self.has(key)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"ConfigManager(file={self._config_file}, keys={len(self._config)})"


def create_default_config() -> Dict[str, Any]:
    """创建智慧农牧业系统的默认配置
    
    返回:
        默认配置字典
    """
    return {
        # 系统基础配置
        "system": {
            "name": "hetao_ag",
            "version": "1.0.0",
            "language": "zh_CN",
            "timezone": "Asia/Shanghai"
        },
        
        # 日志配置
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "logs/hetao_ag.log",
            "max_size_mb": 10,
            "backup_count": 5
        },
        
        # 土壤模块配置
        "soil": {
            "default_depth_m": 0.3,
            "field_capacity": 0.35,
            "wilting_point": 0.12
        },
        
        # 水循环模块配置
        "water": {
            "et_method": "penman_monteith",
            "reference_crop_height_m": 0.12,
            "default_albedo": 0.23
        },
        
        # 作物模块配置
        "crop": {
            "default_growing_season_days": 120,
            "thermal_time_base_celsius": 10.0
        },
        
        # 畜牧模块配置
        "livestock": {
            "detection_confidence": 0.5,
            "model_variant": "yolov5s",
            "use_gpu": True
        },
        
        # 遥感模块配置
        "space": {
            "default_savi_l": 0.5,
            "cloud_mask_threshold": 0.3
        },
        
        # 优化模块配置
        "optimization": {
            "solver": "CBC",
            "max_iterations": 1000,
            "tolerance": 1e-6
        }
    }


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    import tempfile
    
    print("=" * 60)
    print("河套智慧农牧业库 - 配置管理演示")
    print("=" * 60)
    
    # 演示1: 使用默认配置
    print("\n【默认配置】")
    defaults = create_default_config()
    config = ConfigManager(defaults=defaults)
    print(f"  系统名称: {config.get('system.name')}")
    print(f"  日志级别: {config.get('logging.level')}")
    
    # 演示2: 设置和获取配置
    print("\n【动态设置配置】")
    config.set("database.host", "localhost")
    config.set("database.port", 5432)
    print(f"  数据库主机: {config.get('database.host')}")
    print(f"  数据库端口: {config.get('database.port')}")
    
    # 演示3: 环境变量覆盖
    print("\n【环境变量覆盖】")
    os.environ["DATABASE__HOST"] = "production-server"
    print(f"  数据库主机(环境变量覆盖): {config.get('database.host')}")
    del os.environ["DATABASE__HOST"]  # 清理
    
    # 演示4: 保存和加载YAML配置
    if YAML_AVAILABLE:
        print("\n【YAML配置文件】")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.dump({
                'farm': {
                    'name': '河套示范农场',
                    'area_hectares': 500
                }
            }, f, allow_unicode=True)
            temp_path = f.name
        
        file_config = ConfigManager(temp_path)
        print(f"  农场名称: {file_config.get('farm.name')}")
        print(f"  农场面积: {file_config.get('farm.area_hectares')} 公顷")
        
        os.unlink(temp_path)  # 清理临时文件
    
    # 演示5: 获取配置段落
    print("\n【获取配置段落】")
    soil_config = config.get_section("soil")
    print(f"  土壤配置: {soil_config}")
    
    print("\n" + "=" * 60)
    print("配置管理演示完成！")
    print("=" * 60)
