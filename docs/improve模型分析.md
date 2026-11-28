# improve.yaml 模型结构分析与实验建议

> 分析对象：[ultralytics/cfg/models/rt-detr/improve/improve.yaml](../ultralytics/cfg/models/rt-detr/improve/improve.yaml)
> 分析日期：2026-11-19
> 结论速览：640 输入下 **15.29M 参数 / 46.6 GFLOPs**；结构连通性正确，但存在两个会直接卡住训练/部署的硬约束（只支持 640、主干依赖 Triton CUDA kernel），另有若干实验设计问题。下文按"结构 → 数据 → 问题 → 建议"组织。

---

## 1. 模型定位

improve.yaml 是把四个模块组合到 RT-DETR-r18 骨架上的改进模型：

| 组件 | 实现 | 来源 |
| --- | --- | --- |
| 主干 | `Conv` + `C2f_LSBlock` | CVPR2025 LSNet（LSConv） |
| P5 编码器 | `TransformerEncoderLayer_DPB` | CrossFormer（Dynamic Position Bias） |
| 特征融合 | `MFM` 替代 `Concat` | DCMPNet 系列改进 |
| 颈部 | `MANet` 替代 `RepC3` | Hyper-YOLO（HyperACE） |
| 解码器 | `RTDETRDecoder [nc,256,300,4,8,3]` | 与 rtdetr-r18.yaml 一致（3 层解码器） |

可以理解为 `rtdetr-C2f-LSBlock.yaml` 的主干 + `rtdetr-AIFI-DPB.yaml` 的编码器 + `rtdetr-MFM.yaml` 的融合 + `rtdetr-MAN.yaml` 的颈部，`nc` 改为 7。

### 1.1 改进材料与参考文献对应（已核对）

论文材料位于 `/home/pan/Documents/论文材料`，其中 txt 是各改进模块的代码/配置，pdf 是对应参考文献：

| 模块 | 代码材料 | 参考文献 | 论文出处 |
| --- | --- | --- | --- |
| C2f_LSBlock | `LSBlock.txt` | 参考文献3 | LSNet: See Large, Focus Small（CVPR 2025，arXiv:2503.23135） |
| TransformerEncoderLayer_DPB | `DPB.txt` | 参考文献1 | CrossFormer: A Versatile Vision Transformer Hinging on Cross-Scale Attention（ICLR 2022，arXiv:2108.00154） |
| MFM | `MFM.txt` | 参考文献4 | Depth Information Assisted Collaborative Mutual Promotion Network for Single Image Dehazing（DCMPNet，CVPR 2024） |
| MANet | `MFM.txt` | 参考文献2 | Hyper-YOLO: When Visual Object Detection Meets Hypergraph Computation（arXiv:2408.04704，2024） |
| 整体配置 | `改进.txt` | — | 原始 yaml（nc: 9） |

核对结果：

- 三个 txt 中的类实现与仓库 `block.py` / `transformer.py` / `DCMPNet.py` **逐字一致**，材料与代码没有版本偏差；
- `改进.txt` 与 improve.yaml 的 `backbone` / `head` / `scales` **完全一致**，仅 `nc: 9` → `nc: 7`（适配绝缘子数据集的 7 类）；
- 2026-11-22 已清理 improve.yaml：删除四段注释备选配置，层注释编号统一修正为 9–27（与 `改进.txt` 一致）。清理明细见附录 A。
- `LSBlock.txt` 另外包含了 `GatedLSConvBlock_BCHW` 与 `C2f_MambaOut_LSConv`，这两个类未在该方法中使用，属于同源备选模块。

写作提示（方法归属与用法差异）：

- MFM 出自单图像去雾工作 DCMPNet，原文中是 **Modulation Fusion Module（调制融合模块）**，隶属于 LEGM/FMI 结构，用于融合两路恢复特征；这里被抽象成通用的两输入通道门控融合，替代 RT-DETR 的 Concat。
- MANet 出自 Hyper-YOLO，原文的 **Mixed Aggregation Network** 是放在 **backbone** 中做特征提取；这里被移到 **颈部** 替代 RepC3，属于跨位置迁移。
- `LSBlock.txt` 另外附带了 `GatedLSConvBlock_BCHW`、`C2f_MambaOut_LSConv`（MambaOut 版 LSConv），本方法未使用。
- 原始材料 `改进.txt` 为 `nc: 9`，当前仓库适配为 7 类；若论文实验使用的是 9 类版本，需说明配置来源。

论文方法部分建议明确写出"借鉴自 XXX、用于 XXX 位置、做了哪些适配"，避免审稿人对模块原创性与使用位置提出质疑。

## 2. 逐层结构

640×640 输入下（H/W 为对应 stage 的特征图边长）：

| idx | 模块 | 输出通道 | H×W | 备注 |
| --- | --- | --- | --- | --- |
| 0 | Conv k3 s2 | 64 | 320 | P1/2 |
| 1 | Conv k3 s2 | 128 | 160 | P2/4 |
| 2 | C2f_LSBlock | 128 | 160 | |
| 3 | Conv k3 s2 | 256 | 80 | P3/8 |
| 4 | C2f_LSBlock | 256 | 80 | **P3 来源** |
| 5 | Conv k3 s2 | 384 | 40 | P4/16 |
| 6 | C2f_LSBlock | 384 | 40 | **P4 来源** |
| 7 | Conv k3 s2 | 384 | 20 | P5/32 |
| 8 | C2f_LSBlock ×3 | 384 | 20 | **P5** |
| 9 | Conv 1×1 | 256 | 20 | input_proj |
| 10 | TransformerEncoderLayer_DPB | 256 | 20 | 替换 AIFI |
| 11 | Conv 1×1 | 256 | 20 | Y5 lateral |
| 12 | Upsample ×2 | 256 | 40 | |
| 13 | Conv 1×1（from 6） | 256 | 40 | input_proj.1 |
| 14 | MFM（-2, -1） | 256 | 40 | |
| 15 | MANet（n=3, p=2, k=3） | 256 | 40 | fpn_blocks.0 |
| 16 | Conv 1×1 | 256 | 40 | Y4 lateral |
| 17 | Upsample ×2 | 256 | 80 | |
| 18 | Conv 1×1（from 4） | 256 | 80 | input_proj.0 |
| 19 | MFM（-2, -1） | 256 | 80 | |
| 20 | MANet | **128** | 80 | **X3，宽度为其它层一半** |
| 21 | Conv k3 s2 | 256 | 40 | downsample.0 |
| 22 | MFM（-1, 16） | 256 | 40 | |
| 23 | MANet | 256 | 40 | F4 |
| 24 | Conv k3 s2 | 256 | 20 | downsample.1 |
| 25 | MFM（-1, 11） | 256 | 20 | |
| 26 | MANet | 256 | 20 | F5 |
| 27 | RTDETRDecoder | `[7,[128,256,256],256,300,4,8,3]` | — | P3/P4/P5 |

所有 `from` 索引均核对无误（`[6]`=P4、`[4]`=P3、`[16]`=Y4 侧向输出、`[11]`=Y5 侧向输出、解码器 `[[20,23,26]]`）。

> ✅ 2026-11-22 已修正：head 注释编号统一为 9 ~ 27，与 `改进.txt` 和真实索引一致（原先的 `#8` ~ `#25` 来自 8 层主干的 MAN 模板，已随注释块一并清理）。

## 3. 模块说明

### 3.1 C2f_LSBlock
`C2f` 的内部块被替换为 LSNet 的 `Block(depth=1)`（[block.py:11744](../ultralytics/nn/extra_modules/block.py#L11744)），包含：
- `LSConv`：7×7 大核深度卷积 `LKP` 生成逐像素权重，再由 `SKA` 做局部加权求和；
- 残差 FFN（`ed → 2·ed → ed`）。

注意 `SKA` 的实现是 **Triton kernel**（[lsnet.py:126](../ultralytics/nn/backbone/lsnet.py#L126)），不是普通 PyTorch 算子，这带来了第 5.2 节的限制。

### 3.2 TransformerEncoderLayer_DPB
单层全局自注意力 + 动态位置偏置，只作用于 P5（400 token）：

- 注意力维度 = 输入通道 256，8 头；FFN 隐层 = yaml 里的 `[1024]`；
- 参数量 0.79M，与 r18 的 AIFI 持平，但注意力维度从 1024 降到 256；
- `DPB_Attention` 的 `group_size` 在 [transformer.py:993](../ultralytics/nn/extra_modules/transformer.py#L993) 写死为 `(20, 20)`。

### 3.3 MFM
对两路特征求和后做通道门控加权，再按权重混合（[DCMPNet.py:214](../ultralytics/nn/extra_modules/DCMPNet.py#L214)），替代 `Concat`：
- 每个仅 24.6k 参数；
- 输出保持 256 通道（Concat 会翻倍到 512）；
- 代价是下一级无法分别看到两路特征。

### 3.4 MANet
Hyper-YOLO 的多分支块（[block.py:9357](../ultralytics/nn/extra_modules/block.py#L9357)）：
- 1×1 分支、1×1→DWConv→1×1 分支、`chunk(2)` 的两路，再加 n=3 个 Bottleneck；
- 拼接后 `(4+n)·c → c2`，内部宽度 `c = c2·e`，默认 `e=0.5`；
- 256 通道单块约 1.42M 参数，r18 的 RepC3 约 0.66M。

## 4. 参数量与计算量（实测）

在 640×640、fuse 后统计：

| 模型 | 参数量 | GFLOPs@640 |
| --- | --- | --- |
| **improve.yaml** | **15.29M**（15,287,052） | **46.6** |
| improve-method1.yaml（当前默认） | 14.75M | 51.1 |
| rtdetr-r18.yaml | 19.88M | 57.0 |
| rtdetr-l.yaml | 32.15M | 103.8 |

improve.yaml 的参数分布（fuse 前逐层统计）：

| 部分 | 参数量 |
| --- | --- |
| 主干（layer 0–8） | 4.61M |
| 编码器 + 颈部（layer 9–26） | 6.80M |
| 解码器（layer 27） | 3.89M |
| 合计 | 15.30M |

与 r18 对比的增量构成：

- 主干：11.20M → 4.61M（**−6.6M**，主要来自 384 通道的轻量主干）
- 颈部+编码器：4.97M → 6.80M（**+1.8M**，MANet 比 RepC3 重，MFM 很轻）
- 解码器：3.93M → 3.89M（基本持平）

**结论：轻量化收益几乎全部来自主干；颈部实际变重了。** 如果论文以"轻量化"为卖点，这一点需要在文字里讲清楚。

## 5. 已知问题

### 5.1 P0：模型只在 640×640 下能跑

`TransformerEncoderLayer_DPB` 把 group_size 硬编码为 `(20, 20)`，相对位置偏置按 400 个 token 预生成（[transformer.py:924-948](../ultralytics/nn/extra_modules/transformer.py#L924-L948)）。实测：

```
forward 640 ×640 : OK
forward 320 ×320 : RuntimeError: The size of tensor a (100) must match the size of tensor b (400) at non-singleton dimension 3
forward 1280×1280: RuntimeError: The size of tensor a (1600) must match the size of tensor b (400) at non-singleton dimension 3
```

受影响的操作：`train.py --imgsz`、`val.py --imgsz`、`detect.py` 指定其它尺寸、ONNX 动态 shape 导出、多尺度推理/TTA。
（训练与验证的默认 `rect=False`，输入都会 letterbox 到 640 方图，因此默认流程不受影响。）

### 5.2 P0：C2f_LSBlock 依赖 Triton CUDA kernel，兼容性有坑

[lsnet.py:12-20](../ultralytics/nn/backbone/lsnet.py#L12-L20) 先尝试 `from torch.amp import custom_fwd`，`SkaFn` 只在这个 try 成功时才定义；失败时裸 `except` 只定义空壳 `SKA`，**首次 forward 才会报 `NameError: name 'SkaFn' is not defined`**（本机 torch 2.11 实测）。

触发条件：
- torch ≥ 2.4：`torch.amp.custom_fwd` 要求 `device_type` 关键字参数，`@custom_fwd` 直接装饰会抛 TypeError。`requirements.txt` 允许 `torch<2.6`，2.4/2.5 正好中招。
- 无 Triton：同上，`SkaFn` 未定义。
- 无 CUDA / 无 C 编译器：Triton kernel 无法 JIT，forward 抛 `Failed to find C compiler` / CUDA 相关错误。

因此即使前两条都正常：
- **CPU 训练 / 验证 / 推理不可用**（Triton kernel 需要 GPU）；
- **ONNX / TensorRT 导出不可用**（Ultralytics 导出在 CPU 上执行模型）；
- `scripts/check_env.py` 只构建默认模型 improve-method1.yaml 且不做 forward，检查不出以上问题；
- `docs/部署指南.md` 的 CUDA 扩展表格未提及 Triton。

### 5.3 P1：默认模型不是 improve.yaml

[project_paths.py:11](../project_paths.py#L11)、[train.py](../train.py) 与 [README.md](../README.md) 默认使用 `improve-method1.yaml`（方法1，已发表），README 还把 improve.yaml 称为"对照模型"。直接 `python train.py` 训练的是方法1而不是当前方法，需显式指定：

```bash
python train.py --model ultralytics/cfg/models/rt-detr/improve/improve.yaml --name exp_improve
```

建议统一 README、`project_paths.py` 与论文中的"主方法"口径，避免实验记录对不上。

### 5.4 P1：X3 只有 128 通道

r18 的 P3 输出为 256 通道，这里第 20 层 MANet 只给 128，解码器 P3 输入投影也按 128 构建。P3 是最高分辨率、承担小目标检测的层级，而 `broken_disc`、`Glass_Loss` 等缺陷尺寸偏小。建议做 128 vs 256 对照。

### 5.5 P1：无可用预训练权重

主干与官方 RT-DETR / COCO 权重均不兼容，只能从零训练。数据规模为 train 2150 / images/val 391 / images/test 287（注意 `configs/datasets/insulator.yaml` 中 `test` 指向 `images/val`、`val` 指向 `images/test`）。15M 参数 + 7 类小数据集训练 220 epoch，需要关注过拟合与收敛稳定性。

### 5.6 P2：其它

- `scales.l` 实际不起作用（width/depth=1.0），也没有 n/s/m 变体，模型规模只能手改通道。
- 四个改动同时叠加，当前没有消融证据；基线应选 **rtdetr-r18**（解码器深度同为 3），而不是 rtdetr-l。
- MFM 替代 Concat 后，若小目标 AP 不升，应优先做 MFM vs Concat 的对照。

## 6. 修改建议

1. **DPB 参数化**：把 `group_size` 提为构造参数并可用 yaml 传入，或按运行时 H/W 动态生成 `relative_position_index`。最小改动是给 `TransformerEncoderLayer_DPB` 增加 `hw=(20,20)` 参数，由 yaml 显式传入。
2. **LSBlock 兼容性**：
   - 改为 `from torch.cuda.amp import custom_fwd, custom_bwd`（与 requirements 的 torch 范围匹配）；
   - 为 `SKA` 提供纯 PyTorch fallback（unfold/邻域加权），保证 CPU 与导出可用；
   - 在 `requirements.txt` / 部署指南中补充 Triton 与编译器的说明。
3. **修正注释层号**：head 注释整体 +1（9–27）。
4. **统一默认模型**：根据论文主方法修改 `project_paths.DEFAULT_MODEL_YAML` 与 README 描述。
5. **自检脚本加 forward**：`scripts/check_env.py` 增加一次 640 前向冒烟测试，并在报错时提示 Triton/torch 版本问题。

## 7. 消融实验设计

以 `rtdetr-r18.yaml` 为基线（同解码器深度），逐项替换、每次只动一个变量：

| 组 | 主干 | 编码器 | 融合 | 颈部 | X3 宽度 |
| --- | --- | --- | --- | --- | --- |
| A 基线 | ResNet18 | AIFI | Concat | RepC3 | 256 |
| B | **C2f_LSBlock** | AIFI | Concat | RepC3 | 256 |
| C | C2f_LSBlock | **DPB** | Concat | RepC3 | 256 |
| D | C2f_LSBlock | DPB | **MFM** | RepC3 | 256 |
| E | C2f_LSBlock | DPB | MFM | **MANet** | 256 |
| F（当前） | C2f_LSBlock | DPB | MFM | MANet | **128** |
| G | C2f_LSBlock | DPB | MFM | MANet | **256** |

同时建议：
- 报告每组的参数量、GFLOPs、FPS（`tools/get_FPS.py`）与各类 AP，而不只是 mAP50-95；
- 小目标类别单独看 `broken_disc`、`Glass_Loss` 的 AP；
- 如仍不理想，考虑增加 P2 输出层（参考 `rtdetr-p2.yaml`）。

补充：若还要做"MFM 与 MANet 同时替换"的组合对照，可直接使用新增的 `rt-detr/rtdetr-MFM-MANet.yaml`（r18 + AIFI + MFM + MANet，21.94M / 58.7 GFLOPs），无需从 improve.yaml 注释里恢复配置。

## 8. 复现说明

**环境**：Python 3.10、torch 2.11.0+cu128、RTX 2060 6GB（仅用于结构与指标统计，未改动项目环境）。

**统计方式**：`RTDETR(yaml)` 构建后 `fuse()`，用 `ultralytics.utils.torch_utils.model_info` 统计参数量与 GFLOPs。由于本机缺少 C 编译器，Triton SKA kernel 无法 JIT，GFLOPs 统计时用等价的 PyTorch 实现替换了 `SKA.forward`（9 次邻域乘加）；**参数量为精确值，GFLOPs 误差很小**。

**640 前向冒烟测试**：

```bash
python -c "
from ultralytics import RTDETR
import torch
m = RTDETR('ultralytics/cfg/models/rt-detr/improve/improve.yaml')
m.model.eval()
with torch.no_grad():
    m.model(torch.zeros(1, 3, 640, 640))
print('forward 640 OK')
"
```

若报 `NameError: name 'SkaFn' is not defined`，先检查 `torch.__version__`（≥2.4 会触发）与 Triton/CUDA 是否就绪。

---

## 附录 A：improve 目录文件与注释块清理记录

### A.1 文件职责

| 文件 | 内容 | 用途 |
| --- | --- | --- |
| `improve/improve.yaml` | C2f_LSBlock + DPB + MFM + MANet，`nc: 7` | 论文主方法 |
| `improve/improve-method1.yaml` | C2f/C2f_LGLB + LRSA + SNI + RepC3 + GSConvE，`nc: 7` | 方法1（0718 论文，已发表） |
| `improve/rtdetr-AIFI-DyT.yaml` | r18 骨干 + AIFI_DyT + Concat + RepC3 | 单模块对照（`nc: 80`，训练时由数据覆盖） |
| `rt-detr/rtdetr-MFM-MANet.yaml` | r18 骨干 + AIFI + MFM + MANet | 新增：MFM+MANet 组合消融（原 improve.yaml 注释块 1） |

### A.2 improve.yaml 清理明细（2026-11-22）

原文件 216 行，含四段被注释的备选配置。经结构比对后处理如下：

| 原行号 | 内容 | 处理 |
| --- | --- | --- |
| 9–47 | r18 + AIFI + MFM + MANet | 另存为 `rt-detr/rtdetr-MFM-MANet.yaml` |
| 51–87 | LSBlock + DPB + MFM + MANet 旧拷贝 | 删除（与有效配置重复，且有 `c2` / `c2f_LSBlock` 两处笔误） |
| 135–171 | LGLB + LRSA + SNI + RepC3 + GSConvE | 删除（与 `improve-method1.yaml` 完全重复） |
| 177–216 | r18 + AIFI + MFM + RepC3 + SPDConv + CSPOmniKernel | 删除（与 `rtdetr-SOEP-MFM.yaml` 完全重复） |

清理后 improve.yaml 为 49 行，`backbone` / `head` / `scales` 与 `改进.txt` 逐项一致，层注释编号修正为 9–27。清理前后实测指标不变：**15.29M 参数 / 46.6 GFLOPs @640**。

---

## 附录 B：方法1（0718 论文）材料核对

材料目录：`/home/pan/Documents/论文材料/方法1`

| 文件 | 内容 |
| --- | --- |
| `改进1(1).txt` | 方法1 的 yaml（`nc: 9`） |
| `LGLB.txt` | LGLBlock / C2f_LGLB 等模块代码 |
| `LRSA.txt` | LRSA / TransformerEncoderLayer_LRSA |
| `RFPN.txt` | SNI / GSConvE |
| `基于改进RT-DETR的轻量化绝缘子检测0718.docx` | 论文正文 |

模块与仓库实现对照（均为逐字一致）：

| 论文模块 | 类 | 仓库位置 |
| --- | --- | --- |
| C2f_LGLB | `C2f_LGLB` | `ultralytics/nn/extra_modules/block.py` |
| LGLBlock 及其子模块 | `LGLBlock`/`LocalAgg`/`SelfAttn`/`GlobalSparseAttn` | `ultralytics/nn/extra_modules/MobileUViT.py` |
| LRSA-AIFI | `LRSA` / `TransformerEncoderLayer_LRSA` | `ultralytics/nn/extra_modules/transformer.py` |
| SNI 上采样 | `SNI` | `ultralytics/nn/extra_modules/block.py` |
| GSConvE 下采样 | `GSConvE` | `ultralytics/nn/extra_modules/block.py` |
| 颈部融合块 | `RepC3` | 框架自带 |

说明：

- `LGLBlock` 系列不在 `block.py`，而是从 `MobileUViT.py` 导入到 `block.py` 后再被 `C2f_LGLB` 使用；直接搜 `block.py` 会误以为缺失。
- `改进1(1).txt` 与 `improve-method1.yaml` 的 `backbone` / `head` / `scales` 完全一致，仅 `nc: 9` → `nc: 7`。
- 清理时修正了原文件的两处注释：`Y3 (19)` → `X3 (20)`，并补齐 backbone 各层索引。
- 原 `improve-Copy1.yaml` 顶部注释的 r18 + WCA + MFM + MANet 实验块（解码器引用了不存在的第 26 层）已随重命名删除；如需保留可从 git 历史恢复。

新增消融配置实测：`rtdetr-MFM-MANet.yaml` 为 **21.94M 参数 / 58.7 GFLOPs @640**，640 前向正常。
