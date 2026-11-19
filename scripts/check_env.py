#!/usr/bin/env python
"""部署自检：检查依赖、框架、默认模型和数据集是否就绪。

用法（在项目根目录）:
    python scripts/check_env.py
"""
import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_paths import DATA_YAML, DEFAULT_MODEL_YAML  # noqa: E402

REQUIRED = [
    'torch', 'torchvision', 'numpy', 'cv2', 'yaml', 'matplotlib',
    'pandas', 'seaborn', 'tqdm', 'prettytable',
    # 魔改框架导入所需
    'einops', 'timm', 'pytorch_wavelets', 'dill', 'easydict', 'efficientnet_pytorch',
]

errors = []


def ok(msg):
    print(f'  [OK]   {msg}')


def fail(msg):
    print(f'  [FAIL] {msg}')


print('=' * 62)
print('1) Python 与基础依赖')
print('=' * 62)
print(f'  python: {sys.version.split()[0]} ({sys.executable})')
for name in REQUIRED:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, '__version__', '?')
        ok(f'{name} {version}')
    except Exception as e:
        fail(f'{name} 导入失败: {e}')
        errors.append(name)

print()
print('=' * 62)
print('2) GPU / CUDA')
print('=' * 62)
try:
    import torch
    if torch.cuda.is_available():
        ok(f'CUDA 可用，GPU 数量: {torch.cuda.device_count()}，'
           f'当前设备: {torch.cuda.get_device_name(0)}')
        ok(f'torch {torch.__version__} (built with CUDA {torch.version.cuda})')
    else:
        fail('CUDA 不可用（只能用 CPU 跑，训练会非常慢）')
        errors.append('cuda')
except Exception as e:
    fail(f'无法检查 CUDA: {e}')

print()
print('=' * 62)
print('3) ultralytics 魔改框架')
print('=' * 62)
try:
    import ultralytics
    from ultralytics import RTDETR

    ok(f'ultralytics {ultralytics.__version__} 导入成功')
    try:
        RTDETR(str(DEFAULT_MODEL_YAML))
        ok(f'默认模型可构建: {DEFAULT_MODEL_YAML.relative_to(ROOT)}')
    except Exception:
        fail('默认模型构建失败：')
        traceback.print_exc()
        errors.append('model')
except Exception as e:
    fail(f'无法导入 ultralytics: {e}')
    print('       -> 请先把魔改框架拷贝到项目根目录的 ultralytics/，见 docs/部署指南.md 第 2 步')
    errors.append('ultralytics')

print()
print('=' * 62)
print('4) 数据集')
print('=' * 62)
try:
    import yaml

    cfg = yaml.safe_load(DATA_YAML.read_text(encoding='utf-8'))
    base = DATA_YAML.parent  # Ultralytics 以 data yaml 所在目录为基准
    for split in ('train', 'val', 'test'):
        p = (base / cfg[split]).resolve()
        if p.exists() and next(p.iterdir(), None) is not None:
            ok(f'{split}: {p}')
        else:
            fail(f'{split} 不存在或为空: {p}')
            errors.append(f'dataset:{split}')
    expected = {'train', 'val', 'test'}
    if expected.issubset(cfg) and 'names' in cfg:
        ok(f'类别数 nc={cfg.get("nc", len(cfg["names"]))}: {cfg["names"]}')
except Exception as e:
    fail(f'读取数据集配置失败: {e}')
    errors.append('data-yaml')

print()
print('=' * 62)
if errors:
    print(f'自检发现 {len(errors)} 个问题: {", ".join(errors)}')
    print('请对照 docs/部署指南.md 处理后重新运行本脚本。')
    sys.exit(1)
print('环境自检全部通过，可以开始训练: python train.py')
