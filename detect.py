import argparse
import warnings

warnings.filterwarnings('ignore')

from ultralytics import RTDETR

from project_paths import DEFAULT_WEIGHTS, ROOT

# 预测框粗细和颜色修改问题可参考原项目的《使用教程.md》。


def parse_args():
    parser = argparse.ArgumentParser(description='用训练好的权重做推理')
    parser.add_argument('--weights', default=str(DEFAULT_WEIGHTS), help='权重路径')
    parser.add_argument('--source', default=str(ROOT / 'datasets/insulator/images/test'), help='图片/视频/目录')
    parser.add_argument('--conf', type=float, default=0.25)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = RTDETR(args.weights)  # select your model.pt path
    model.predict(source=args.source,
                  conf=args.conf,
                  project=str(ROOT / 'runs/detect'),
                  name='exp',
                  save=True,
                  # visualize=True,  # visualize model feature maps
                  # line_width=2,  # line width of the bounding boxes
                  # show_conf=False,  # do not show prediction confidence
                  # show_labels=False,  # do not show prediction labels
                  # save_txt=True,  # save results as .txt file
                  # save_crop=True,  # save cropped images with results
                  )
