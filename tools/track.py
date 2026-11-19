import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import ROOT

import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR(str(ROOT / 'weights/rtdetr-r18.pt')) # select your model.pt path
    model.track(source=str(ROOT / 'video.mp4'),
                project=str(ROOT / 'runs/track'),
                name='exp',
                save=True
                )