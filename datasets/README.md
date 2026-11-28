# 数据集说明

本目录不入库（`.gitignore` 已忽略），每个新服务器需要单独传输。

> 两个数据集的 `insulator` 语义不同、7 类实际只有 6 类有实例、类别关系混合粒度，
> 详细分析和 6 类派生方案见 [`docs/数据集问题与对齐方案.md`](../docs/数据集问题与对齐方案.md)。

本机有两个数据集，实体都放在 `/home/pan/datasets/`，项目内通过软链接访问：

| 项目内路径 | 实体路径 | 类别数 | 图像数 |
| --- | --- | --- | --- |
| `datasets/insulator` | `/home/pan/datasets/insulator` | 7 | 2828 |
| `datasets/power-defects-9cls` | `/home/pan/datasets/power-defects-9cls` | 9 | 7972 |

## 1. 绝缘子数据集（7 类）

```
datasets/insulator/
├── images/
│   ├── train/     # 2150 张
│   ├── val/       # 391 张（data.yaml 中映射为 test）
│   └── test/      # 287 张（data.yaml 中映射为 val）
└── labels/
    ├── train/
    ├── val/
    └── test/
```

配置见 [`configs/datasets/insulator.yaml`](../configs/datasets/insulator.yaml)。

> 注意：split 映射沿用原始项目：`test -> images/val`、`val -> images/test`。

来源：Roboflow Insulator Faults Detection
<https://universe.roboflow.com/project-vmgqx/insulator-faults-detection>

## 2. 电力缺陷数据集（9 类）

```
datasets/power-defects-9cls/
├── images/{train,val,test}/   # 5543 / 1595 / 834
├── labels/{train,val,test}/
├── classes.txt
└── data.yaml
```

配置见 [`configs/datasets/power-defects-9cls.yaml`](../configs/datasets/power-defects-9cls.yaml)。

split 直接对应 `images/{train,val,test}`，没有反向映射。类别：

`insulator`、`insulator_stringdrop`、`insulator_breakage`、`insulator_flashover`、
`damper`、`damper_defect`、`nest`、`junyahuan`、`junyahuanqingxie`

## 本地开发

两个数据集的实体都放在 `/home/pan/datasets/`，不要再依赖
`/home/pan/project/RTDETR-20260410`（避免旧项目被误删）。项目内软链接：

```bash
datasets/insulator          -> /home/pan/datasets/insulator
datasets/power-defects-9cls -> /home/pan/datasets/power-defects-9cls
```

新服务器上请把真实目录拷贝到 `datasets/` 下，或重建同样的软链接。
