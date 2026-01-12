from pathlib import Path
import logging
import os
from datetime import datetime

def setup_logger(save_path):
    os.makedirs(save_path / 'logs', exist_ok=True)
    log_filename = f"{save_path}/logs/log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    print("Logging to:", log_filename)

    logging.basicConfig(
        filename=log_filename,
        filemode="w",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    return logging.getLogger(__name__)


def mark_undone(path: Path, step: str):
    try:
        (path / f".{step}_done").unlink()
    except FileNotFoundError:
        pass


def mark_done(path: Path, step: str):
    (path / f".{step}_done").touch()


def is_step_done(path: Path, step: str):
    return (path / f".{step}_done").exists()


def format_feature_names(original, add_params=True):
    feat_names = []
    for feat_name in original:
        first = feat_name.replace('value__', '')
        second = first.split('__')
        feature = second[0]
        params = second[1:]

        if params and add_params:
            string = '('
            for param in params:
                var, value = param.rsplit('_', 1)
                string += f'{var}={value}, '
            string = string[:-2] + ')'
            feature += string
        feat_names.append(feature)
    return feat_names
