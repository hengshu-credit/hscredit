"""日志工具.

提供统一的日志配置和管理功能。
"""

import os
import sys
import logging
from datetime import datetime


def init_logger(
    name: str = "hscredit",
    level: int = logging.INFO,
    log_file: str = None,
    format: str = None,
    console: bool = True,
) -> logging.Logger:
    """初始化日志记录器。

    :param name: logger 名称，默认为 "hscredit"
    :param level: 日志级别，默认为 logging.INFO
    :param log_file: 日志文件路径，默认为 None（不写入文件）
    :param format: 日志格式，默认为 "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    :param console: 是否输出到控制台，默认为 True
    :return: 配置好的 logger 对象
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # 默认格式
    if format is None:
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format)
    
    # 控制台输出
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件输出
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "hscredit") -> logging.Logger:
    """获取已存在的 logger。

    :param name: logger 名称
    :return: logger 对象
    """
    return logging.getLogger(name)
