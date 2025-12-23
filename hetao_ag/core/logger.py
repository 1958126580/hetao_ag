# -*- coding: utf-8 -*-
"""
hetao_ag.core.logger - 日志工具模块

提供标准化的日志功能，支持控制台和文件输出。

作者: Hetao College
版本: 1.0.0
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

try:
    from logging.handlers import RotatingFileHandler
    HANDLERS_AVAILABLE = True
except ImportError:
    HANDLERS_AVAILABLE = False


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    COLORS = {
        'DEBUG': '\033[90m',
        'INFO': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'CRITICAL': '\033[41m',
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        message = super().format(record)
        if sys.stdout.isatty() and color:
            return f"{color}{message}{self.RESET}"
        return message


class Logger:
    """智慧农牧业日志记录器
    
    示例:
        >>> logger = Logger("hetao_ag.soil")
        >>> logger.info("开始土壤分析...")
    """
    
    DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    _loggers: dict = {}
    
    def __init__(
        self,
        name: str = "hetao_ag",
        level: Union[str, int] = logging.INFO,
        log_file: Optional[Union[str, Path]] = None,
        console_output: bool = True,
        colored: bool = True
    ):
        self.name = name
        
        if name in Logger._loggers:
            self._logger = Logger._loggers[name]
        else:
            self._logger = logging.getLogger(name)
            Logger._loggers[name] = self._logger
        
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        self._logger.setLevel(level)
        self._logger.handlers.clear()
        
        if console_output:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level)
            if colored:
                formatter = ColoredFormatter(self.DEFAULT_FORMAT, datefmt=self.DEFAULT_DATE_FORMAT)
            else:
                formatter = logging.Formatter(self.DEFAULT_FORMAT, datefmt=self.DEFAULT_DATE_FORMAT)
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
        
        if log_file:
            self.add_file_handler(log_file)
    
    def add_file_handler(self, log_file: Union[str, Path], max_size_mb: float = 10.0, backup_count: int = 5):
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if HANDLERS_AVAILABLE:
            handler = RotatingFileHandler(log_path, maxBytes=int(max_size_mb * 1024 * 1024), backupCount=backup_count, encoding='utf-8')
        else:
            handler = logging.FileHandler(log_path, encoding='utf-8')
        
        handler.setLevel(self._logger.level)
        handler.setFormatter(logging.Formatter(self.DEFAULT_FORMAT, datefmt=self.DEFAULT_DATE_FORMAT))
        self._logger.addHandler(handler)
    
    def set_level(self, level: Union[str, int]):
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        self._logger.setLevel(level)
        for handler in self._logger.handlers:
            handler.setLevel(level)
    
    def debug(self, msg: str, **kwargs):
        extra = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        self._logger.debug(msg + (" | " + extra if extra else ""))
    
    def info(self, msg: str, **kwargs):
        extra = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        self._logger.info(msg + (" | " + extra if extra else ""))
    
    def warning(self, msg: str, **kwargs):
        extra = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        self._logger.warning(msg + (" | " + extra if extra else ""))
    
    def error(self, msg: str, exc_info: bool = False, **kwargs):
        extra = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        self._logger.error(msg + (" | " + extra if extra else ""), exc_info=exc_info)
    
    def critical(self, msg: str, exc_info: bool = True, **kwargs):
        extra = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        self._logger.critical(msg + (" | " + extra if extra else ""), exc_info=exc_info)
    
    def exception(self, msg: str, **kwargs):
        extra = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        self._logger.exception(msg + (" | " + extra if extra else ""))
    
    def log_experiment_start(self, experiment_name: str, parameters: Optional[dict] = None, random_seed: Optional[int] = None):
        """记录实验开始"""
        self.info("=" * 50)
        self.info(f"实验开始: {experiment_name}")
        self.info(f"时间: {datetime.now().isoformat()}")
        if random_seed is not None:
            self.info(f"随机种子: {random_seed}")
        if parameters:
            for key, value in parameters.items():
                self.info(f"  {key}: {value}")
        self.info("=" * 50)
    
    def log_experiment_end(self, success: bool = True, results: Optional[dict] = None):
        """记录实验结束"""
        self.info("=" * 50)
        self.info(f"实验结束 - 状态: {'成功' if success else '失败'}")
        if results:
            for key, value in results.items():
                self.info(f"  {key}: {value}")
        self.info("=" * 50)


def get_logger(name: str = "hetao_ag", **kwargs) -> Logger:
    """获取日志记录器"""
    return Logger(name, **kwargs)


if __name__ == "__main__":
    logger = get_logger("hetao_ag.demo", level="DEBUG")
    logger.debug("调试信息")
    logger.info("系统初始化完成")
    logger.warning("内存使用率高", memory_percent=85)
