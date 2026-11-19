# 数据集说明

本目录不入库（`.gitignore` 已忽略），每个新服务器需要单独传输。

## 目录结构

```
datasets/insulator/
├── images/
│   ├── train/     # 训练集图片
│   ├── val/       # 验证集图片（data.yaml 中映射为 test）
│   └── test/      # 测试集图片（data.yaml 中映射为 val）
└── labels/
    ├── train/     # YOLO 格式标签
    ├── val/
    └── test/
```

数据集配置见 [`configs/datasets/insulator.yaml`](../configs/datasets/insulator.yaml)（该文件入库）。

## 来源

Roboflow: Insulator Faults Detection（Public Domain）
<https://universe.roboflow.com/project-vmgqx/insulator-faults-detection>

## 本地开发

本机为了不重复占用 185M，`datasets/insulator` 是指向旧项目
`../../RTDETR-20260410/dataset` 的软链接；新服务器上请拷贝真实目录。
