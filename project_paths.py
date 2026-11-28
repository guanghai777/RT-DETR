"""项目公共路径。

所有脚本都从这里取路径，这样仓库移动到任意服务器都不需要改代码。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 模型配置（本仓库管理）
MODEL_DIR = ROOT / 'ultralytics/cfg/models/rt-detr'
DEFAULT_MODEL_YAML = MODEL_DIR / 'improve/improve-method1.yaml'

# 数据集配置与数据（数据本身不入库）
DATA_YAML = ROOT / 'configs/datasets/insulator.yaml'
DATASET_DIR = ROOT / 'datasets/insulator'

# 输出
RUNS_DIR = ROOT / 'runs'
TRAIN_RUN_DIR = RUNS_DIR / 'train/exp'
DEFAULT_WEIGHTS = TRAIN_RUN_DIR / 'weights/best.pt'
WEIGHTS_DIR = ROOT / 'weights'
