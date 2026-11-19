# 绝缘子缺陷检测 · RT-DETR 改进工程

基于 ultralytics 8.0.201 魔改框架的 RT-DETR 改进项目，面向绝缘子 7 类缺陷检测。

- **任务**：目标检测（7 类：`insulator`、`110_two_hight_glass`、`broken_disc`、
  `Glass_Dirty`、`Glass_Loss`、`pollution-flashover`、`Polyme_Dirty`）
- **数据集**：Roboflow Insulator Faults Detection，共约 185M
- **默认模型**：`ultralytics/cfg/models/rt-detr/improve/improve-Copy1.yaml`
  （C2f_LGLB + TransformerEncoderLayer_LRSA + SNI + RepC3 + GSConvE）
- **对照模型**：`improve/improve.yaml`（C2f_LSBlock + DPB + MFM + MANet）

## 目录结构

```
rtdetr-improve/
├── train.py / val.py / detect.py / export.py   # 入口脚本
├── project_paths.py                            # 统一路径定义（换服务器无需改代码）
├── configs/
│   └── datasets/insulator.yaml                 # 数据集配置（入库）
├── ultralytics/
│   └── cfg/models/                             # ★ 改进模型 yaml（入库）
│       ├── rt-detr/improve/                    #   本项目的改进配置
│       ├── rt-detr/                            #   模块组合对照 yaml
│       └── yolo-detr/
├── tools/                                      # 分析脚本（FPS、热力图、ERF、COCO 指标…）
├── scripts/
│   ├── setup_env.sh                            # 一键安装依赖
│   └── check_env.py                            # 部署自检
├── docs/
│   ├── 部署指南.md                              # ★ 新服务器快速部署
│   └── LOSS改进系列.md
├── datasets/                                   # 数据集（不入库）
├── weights/                                    # 权重（不入库）
└── runs/                                       # 训练/验证输出（不入库）
```

## 版本管理说明

**入库**：全部脚本、`configs/`、`docs/`、`ultralytics/cfg/models/`（所有改进与对照 yaml）。

**不入库**（`.gitignore`）：

| 内容 | 原因 | 如何获取 |
| --- | --- | --- |
| `ultralytics/` 框架源码（除 `cfg/models`） | 22M 魔改框架，随环境部署 | [部署指南 · 第 2 步](docs/部署指南.md) |
| `datasets/` | 185M，含第三方数据授权 | [部署指南 · 第 4 步](docs/部署指南.md) |
| `weights/` `*.pt` `*.onnx` | 大文件 | [weights/README.md](weights/README.md) |
| `runs/` | 训练产物 | 本地训练生成 |

> 注意：`ultralytics/nn/` 里的自定义模块实现来自原始魔改框架，不在本仓库版本控制范围内。
> 新机器上必须先把完整框架拷贝到项目根目录，否则模型无法构建。

## 快速开始（框架与数据已就绪的机器）

```bash
# 0) 自检
python scripts/check_env.py

# 1) 训练（默认 improve-Copy1.yaml）
python train.py

# 2) 验证并输出论文指标表（参数量 / GFLOPs / FPS / 各类 mAP）
python val.py

# 3) 推理
python detect.py --source datasets/insulator/images/test
```

更多参数：

```bash
python train.py --help
python train.py --model ultralytics/cfg/models/rt-detr/improve/improve.yaml --name exp_manet
```

## 部署到新服务器

见 **[docs/部署指南.md](docs/部署指南.md)**，包含框架/数据集传输命令、环境安装、
CUDA 扩展、自检清单与常见问题。

## 数据与权重

- 数据集结构与来源：[datasets/README.md](datasets/README.md)
- 预训练权重下载：[weights/README.md](weights/README.md)
- 注意：数据集中 `test` 映射到 `images/val`、`val` 映射到 `images/test`（沿用原项目，
  `configs/datasets/insulator.yaml` 内有说明）

## 工具脚本

| 脚本 | 用途 |
| --- | --- |
| `val.py`（根目录） | 验证 + 论文指标表 |
| `tools/get_FPS.py` | 测推理 FPS / 延迟 |
| `tools/get_model_erf.py` | 有效感受野 ERF |
| `tools/heatmap.py` | Grad-CAM 系列热力图（需 `requirements-tools.txt`） |
| `tools/get_COCO_metrice.py` | COCO 指标（pycocotools + TIDE） |
| `tools/get_all_yaml_param_and_flops.py` | 批量统计各 yaml 的参数量 / FLOPs |
| `tools/main_profile.py` | 模型结构 profile |
| `tools/plot_result.py` | 多条训练曲线对比 |
| `tools/voc-yolo.py` | VOC XML → YOLO txt |
| `tools/test_env.py` | CUDA 扩展安装自检 |

## 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics)（AGPL-3.0）
- 原始 RT-DETR 改进框架与教程
- 数据集：Roboflow *Insulator Faults Detection*
