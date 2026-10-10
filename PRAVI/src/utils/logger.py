# coding: utf-8
# @email: jinfeng.xu0605@gmail.com / jinfeng@connect.hku.hk

import logging
import os
from utils.utils import get_local_time


def init_logger(config):
    """
    Initialize logging system
    
    Functions:
    1. Create subfolders by model-dataset: log/{model}/{dataset}/
    2. Log filename format: {model}-{dataset}-{timestamp}.log
    3. Support colored output (terminal)
    4. Support training visualization toggle
    """
    # Organize log folders by model-dataset
    LOGROOT = './log/'
    model_dataset_dir = os.path.join(LOGROOT, config['model'], config['dataset'])
    
    if not os.path.exists(model_dataset_dir):
        os.makedirs(model_dataset_dir)

    # Log filename
    logfilename = '{}-{}-{}.log'.format(
        config['model'], 
        config['dataset'], 
        get_local_time().replace(' ', '-').replace(':', '-')
    )

    logfilepath = os.path.join(model_dataset_dir, logfilename)

    # Terminal output format (with colors)
    sfmt = u"%(asctime)-15s %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = logging.Formatter(sfmt, sdatefmt)
    
    # File output format (detailed)
    filefmt = "%(asctime)-15s %(levelname)-8s %(message)s"
    filedatefmt = "%Y-%m-%d %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    # Set log level
    if config.get('state') is None or config.get('state').lower() == 'info':
        level = logging.INFO
    elif config.get('state').lower() == 'debug':
        level = logging.DEBUG
    elif config.get('state').lower() == 'error':
        level = logging.ERROR
    elif config.get('state').lower() == 'warning':
        level = logging.WARNING
    elif config.get('state').lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    # File handler
    fh = logging.FileHandler(logfilepath, 'w', 'utf-8')
    fh.setLevel(level)
    fh.setFormatter(fileformatter)

    # Terminal handler
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=[sh, fh]
    )
    
    # Print log path information
    logging.info(f"📁 Log directory: {model_dataset_dir}")
    logging.info(f"📄 Log file: {logfilename}")
    logging.info(f"📊 Visualization: {'Enabled' if config.get('enable_visualization', False) else 'Disabled'}")