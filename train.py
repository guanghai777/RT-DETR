import argparse
import warnings

warnings.filterwarnings('ignore')

from ultralytics import RTDETR

from project_paths import DEFAULT_MODEL_YAML, DATA_YAML, ROOT


def parse_args():
    parser = argparse.ArgumentParser(description='训练 RT-DETR 改进模型')
    parser.add_argument('--model', default=str(DEFAULT_MODEL_YAML), help='模型 yaml 路径')
    parser.add_argument('--data', default=str(DATA_YAML), help='数据集 yaml 路径')
    parser.add_argument('--epochs', type=int, default=220)
    parser.add_argument('--batch', type=int, default=16, help='batchsize 不建议乱动，4 左右通常效果最好')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int, default=4, help='Windows 卡死时设为 0')
    parser.add_argument('--device', default=None, help="如 '0'、'0,1'；不填则自动选择")
    parser.add_argument('--name', default='exp', help='输出目录名，位于 runs/train/ 下')
    parser.add_argument('--resume', default=None, help='断点续训的 last.pt 路径')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # 现版本保存的权重是 fp32（旧版为 fp16，fp16 在 val 时可能出现精度为 0 的问题）
    model = RTDETR(args.model)
    # model.load('')  # 加载预训练权重时填入权重路径

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=0,  # 0 表示不早停
        project=str(ROOT / 'runs/train'),
        name=args.name,
        resume=args.resume,
    )
