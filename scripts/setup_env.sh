#!/usr/bin/env bash
# 一键安装基础环境（推荐在 conda/venv 虚拟环境中执行）
#
# 用法:
#   bash scripts/setup_env.sh                # 仅基础依赖
#   bash scripts/setup_env.sh --with-tools   # 额外安装 tools/ 分析脚本依赖
#   PIP_MIRROR=https://mirrors.aliyun.com/pypi/simple bash scripts/setup_env.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo ">> 项目目录: $(pwd)"
echo ">> Python: $(python -V 2>&1)"
echo ">> pip 镜像: $PIP_MIRROR"

if python -c "import torch" >/dev/null 2>&1; then
    TORCH_VER="$(python -c 'import torch; print(torch.__version__)')"
    echo ">> 检测到已安装 torch ${TORCH_VER}，跳过 torch/torchvision（如版本不匹配请手动处理）"
    TMP_REQ="$(mktemp)"
    grep -viE '^[[:space:]]*(torch|torchvision)([<>=]|$)' requirements.txt > "${TMP_REQ}"
    python -m pip install -r "${TMP_REQ}" -i "${PIP_MIRROR}"
    rm -f "${TMP_REQ}"
else
    echo ">> 未检测到 torch，按 requirements.txt 安装"
    python -m pip install -r requirements.txt -i "${PIP_MIRROR}"
fi

if [[ "${1:-}" == "--with-tools" ]]; then
    echo ">> 安装 tools/ 可选依赖"
    python -m pip install -r requirements-tools.txt -i "${PIP_MIRROR}"
fi

echo
echo ">> 开始环境自检"
python scripts/check_env.py
