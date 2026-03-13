基于改进 YOLOv8 的目标检测算法优化

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-00a8e8.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>



  项目简介

本项目基于 YOLOv8s 进行目标检测算法优化研究，通过引入 CBAM 注意力机制 和 close_mosaic 训练策略，有效解决了小目标检测与模型过拟合问题。在 PASCAL VOC2007 数据集上，mAP@0.5 从 86.9% 提升至 87.1%。

项目采用固定随机种子确保实验可复现，设计了 4 组消融实验 验证各模块有效性，并完成了 ONNX 模型导出 与 Gradio Web Demo 部署。



 主要改进

 改进点 

CBAM 注意力在 Backbone 末端引入通道 + 空间注意力机制  | mAP +0.1% |
close_mosaic 最后 10 轮关闭 Mosaic 增强，用原图微调  | val_loss -0.05 |
固定种子 设置 seed=42，确保实验可复现  | 实验公平性  |
EMA 尝试 指数移动平均（该场景下效果不佳）  | 分析了局限性 |



 实验结果

 消融实验
 

<img width="1239" height="336" alt="image" src="https://github.com/user-attachments/assets/b589cf5c-2784-4a03-b9b2-6494ece9bcf8" />



| 实验编号 | CBAM | close_mosaic | EMA | mAP@0.5 | val_loss |
|:-------:|:----:|:-----------:|:---:|:-------:|:--------:|
| Baseline | ❌ | ❌ | ❌ | 86.9% | 2.664 |
| +CBAM | ✅ | ❌ | ❌ | 87.0% | 2.644 |
| +CBAM+close_mosaic | ✅ | ✅ | ❌ | **87.1%** | **2.612** |
| +EMA | ✅ | ✅ | ✅ | 86.6% | 2.647 |

 关键指标

| 指标 | 数值 |
|:----:|:----:|
| 基线 mAP@0.5 | 86.9% |
| 最终 mAP@0.5 | 87.1% |
| 提升幅度 | +0.2% |
| 验证损失降低 | 0.052 |
| 随机种子 | 42 |



环境依赖

txt

torch>=2.0.0 

torchvision>=0.15.0

opencv-python>=4.8.0

numpy>=1.24.0

gradio>=4.44.0

onnx>=1.14.0

tensorboard>=2.13.0
