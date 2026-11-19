import argparse
import warnings

warnings.filterwarnings('ignore')

from ultralytics import RTDETR

from project_paths import DEFAULT_WEIGHTS

# onnx onnxsim onnxruntime onnxruntime-gpu
# 导出参数官方详解：https://docs.ultralytics.com/modes/export/#usage-examples


def parse_args():
    parser = argparse.ArgumentParser(description='导出 ONNX / TensorRT 模型')
    parser.add_argument('--weights', default=str(DEFAULT_WEIGHTS), help='权重路径或模型 yaml 路径')
    parser.add_argument('--format', default='onnx', choices=['onnx', 'engine', 'torchscript', 'openvino'])
    parser.add_argument('--simplify', action='store_true', default=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = RTDETR(args.weights)
    model.export(format=args.format, simplify=args.simplify)
    # 导出的 TensorRT engine 需使用官方 Ultralytics 做推理，本项目的 detect.py 不支持
