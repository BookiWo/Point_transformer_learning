#!/bin/bash
# ============================================================
# AutoDL 训练准备脚本 — 打包数据 + 验证完整性
# ============================================================
# 运行: bash scripts/prepare_autodl.sh
# 输出: autodl_data_package.tar.gz (~400MB, 含 processed + splits)
# 上传: 用 AutoDL 的 "文件上传" 或 scp 传至 /root/autodl-tmp/
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."

PACK="autodl_data_package.tar.gz"
NOW=$(date '+%Y-%m-%d %H:%M:%S')

echo "============================================"
echo "  AutoDL Data Package Builder"
echo "  $NOW"
echo "============================================"
echo ""

# 1. 验证必需目录
echo "[1/4] Verifying data directories..."
for d in datasets/processed/partnet_pt datasets/splits; do
    if [ ! -d "$d" ]; then
        echo "  ERROR: $d not found!"
        exit 1
    fi
done
echo "  ✅ datasets/processed/ + datasets/splits/ 存在"

# 2. 计算大小
echo ""
echo "[2/4] Estimating package size..."
SPLIT_SIZE=$(du -sh datasets/splits 2>/dev/null | cut -f1)
PROC_SIZE=$(du -sh datasets/processed 2>/dev/null | cut -f1)
echo "  splits:    $SPLIT_SIZE"
echo "  processed: $PROC_SIZE"
echo "  (datasets/raw/ 不需要 — 除非 re-preprocess)"

# 3. 打包
echo ""
echo "[3/4] Building $PACK ..."
tar -czf "$PACK" \
    datasets/processed/partnet_pt \
    datasets/splits/ \
    2>/dev/null

PKG_SIZE=$(du -sh "$PACK" 2>/dev/null | cut -f1)
echo "  ✅ $PACK created ($PKG_SIZE)"

# 4. 校验
echo ""
echo "[4/4] Verifying archive..."
tar -tzf "$PACK" | head -5
TOTAL_FILES=$(tar -tzf "$PACK" | wc -l)
echo "  ... 共 $TOTAL_FILES 个文件"

echo ""
echo "============================================"
echo "  Upload Instructions"
echo "============================================"
echo ""
echo "  AutoDL 控制台 → 我的容器 → 文件上传"
echo "  目标路径: /root/autodl-tmp/"
echo ""
echo "  上传后解压:"
echo "    cd /root/autodl-tmp"
echo "    tar -xzf autodl_data_package.tar.gz"
echo ""
echo "  或者使用 AutoDL 网盘 (更快):"
echo "    cp autodl_data_package.tar.gz /root/autodl-tmp/"
echo ""
echo "============================================"
echo "  Done! Package: $PACK ($PKG_SIZE)"
echo "============================================"
