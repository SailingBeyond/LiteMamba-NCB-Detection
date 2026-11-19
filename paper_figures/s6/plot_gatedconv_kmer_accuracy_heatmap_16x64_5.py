#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补充材料：两个五分类 GatedConv 逐背景热图。

直接运行 python plot_gatedconv_kmer_accuracy_heatmap_16x64_5.py，或以 --result-root 指定统一结果目录。
完整参数：python plot_gatedconv_kmer_accuracy_heatmap_16x64_5.py --help
本入口不训练模型、不重新执行分类推理。
"""
from plot_common import main

if __name__ == "__main__":
    raise SystemExit(main("heatmap5"))
