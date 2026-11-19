# 预训练 / 训练权重

权重文件不入库，训练产物默认在 `runs/train/<name>/weights/`。

## 预训练权重下载

- 百度网盘：<https://pan.baidu.com/s/1Hdy-qLt91jr_FNpkKjPV_w?pwd=dzny> 提取码 `dzny`
  （`rtdetr-weights-new.zip`，含 rtdetr-r18 等基础权重）

把需要的 `.pt` 放到本目录，`tools/track.py`、`tools/heatmap.py` 默认使用
`weights/rtdetr-r18.pt`。
