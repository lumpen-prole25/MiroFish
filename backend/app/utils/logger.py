"""
Logger Configuration模块
提供统one's log管理, simultaneouslyoutputto 控制台 and file
"""

import os
import sys
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler


def _ensure_utf8_stdout():
    """
    ensure stdout/stderr use UTF-8 编码
    解决 Windows 控制台文乱码question
    """
    if sys.platform == 'win32':
        # Windows 下重新configuration标准output UTF-8
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# Loggingdirectory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')


def setup_logger(name: str = 'mirofish', level: int = logging.DEBUG) -> logging.Logger:
    """
    setuplog器
    
    Args:
        name: log器name
        level: Log level
        
    Returns:
        configuration好's log器
    """
    # Ensurelogdirectoryexists
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Createlog器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 阻止log向uploading播to 根 logger, 避免重复output
    logger.propagate = False
    
    # If已经hasprocess器, 不重复add
    if logger.handlers:
        return logger
    
    # Loggingformat
    detailed_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 1. fileprocess器 - 详细log(by date命name, 带round转)
    log_filename = datetime.now().strftime('%Y-%m-%d') + '.log'
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, log_filename),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # 2. 控制台process器 - 简洁log(INFO and 以上)
    # Ensure Windows 下use UTF-8 编码, 避免文乱码
    _ensure_utf8_stdout()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # addprocess器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = 'mirofish') -> logging.Logger:
    """
    Getlog器(如果does not exist则create)
    
    Args:
        name: log器name
        
    Returns:
        log器instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


# Createdefaultlog器
logger = setup_logger()


# 便捷method
def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs)

def critical(msg, *args, **kwargs):
    logger.critical(msg, *args, **kwargs)

