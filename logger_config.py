import logging
import sys


def setup_logger(name="data_prep", level=logging.INFO):
    """
    สร้าง logger สำหรับ data preparation pipeline
    
    Args:
        name: ชื่อ logger
        level: logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # ป้องกันการสร้าง handler ซ้ำ
    if logger.handlers:
        return logger
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Format: timestamp - logger_name - level - message
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger
