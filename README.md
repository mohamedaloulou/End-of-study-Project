# How Effective is Pre-training of Large Masked Autoencoders for Downstream Earth Observation Tasks? (BMVC workshop 2024)

## Updates
- **September 27, 2024:** the paper is released [[arXiv]]([https://arxiv.org/abs/2403.05419](https://arxiv.org/abs/2409.18536))  [[PDF]]([https://arxiv.org/pdf/2403.05419.pdf](https://arxiv.org/pdf/2409.18536))
- **Nov 14, 2024:** Codebase is released.

## Overview
Self-supervised pre-training has proven highly effective for many computer vision tasks, particularly when labelled data are scarce. In the context of Earth Observation (EO), foundation models and various other Vision Transformer (ViT)-based approaches have been successfully applied for transfer learning to downstreamtasks. However, it remains unclear under which conditions pre-trained models
offer significant advantages over training from scratch. In this study, we investigate the effectiveness of pre-training ViT-based Masked Autoencoders (MAE) for downstream EO tasks, focusing on reconstruction, segmentation, and classification. We consider two large ViT-based MAE pre-trained models: a foundation model (Prithvi) and SatMAE. We evaluate Prithvi on reconstruction and segmentation-
based downstream tasks, and for SatMAE we assess its performance on a classification downstream task. Our findings suggest that pre-training is particularly beneficial when the fine-tuning task closely resembles the pre-training task, e.g.reconstruction. In contrast, for tasks such as segmentation or classification, training from scratch with specific hyperparameter adjustments proved to be equally or more effective.
## Proposed Study
We analyze two settings: Setting 1 initializes the encoder E with pre-trained weights from a self-supervised pre-training stage, then fine-tunes it with a task-specific model Mi using supervised learning, while Setting 2 omits the pre-training stage and trains E plus Mi from scratch, comparing the task-specific metrics for both settings.

<img width="1096" alt="image" src="images/overall_architecture.png">
