# Exploration Log: Train Framework LoRA Minimal

## Status: under_review

## Context
`fastvideo/train/` 是当前推荐的训练框架，但此前 LoRA 训练能力主要停留在
`fastvideo/training/` 旧 pipeline 中。为了给 YAML 驱动的新框架补一个最小
LoRA finetune 路径，这次实现重点放在 model plugin 侧的 LoRA 注入，而不是
复制旧 pipeline 的整套训练逻辑。

## Progress
- [x] 阅读训练框架、LoRA 推理层和旧训练 pipeline 的实现。
- [x] 识别新框架缺口：缺少训练侧 LoRA 层注入入口。
- [x] 在 `fastvideo/train/utils/lora.py` 中补充最小注入逻辑。
- [x] 将 LoRA 参数接入 `WanModel` / `HunyuanModel` / `WanCausalModel`。
- [x] 补充 YAML 最小示例和文档说明。
- [x] 将 `WanModel` 的新训练栈从纯 T2V batch 扩到支持 I2V/TI2V parquet。
- [x] 补充 `Wan2.2 TI2V 5B` 的最小 LoRA YAML 示例和轻量回归测试。
- [x] 为 `Matrix-Game-2.0` family 补充新训练栈 model plugin、action-conditioned
  validation 透传和 LoRA YAML 示例。
- [x] 复核 `GEN3C-Cosmos-7B` 在当前仓库中的训练可行性，确认它仍停留在推理集成，
  尚未接入 `fastvideo/train/` LoRA finetune 路径。

## Findings
- 推理侧 `LoRAPipeline` 已经具备可复用的 LoRA layer 封装，训练侧真正缺的是
  “何时把 transformer 替换为 LoRA layer”。
- 新框架的模型加载阶段已经完成 FSDP/HSDP 分片，因此新增的 LoRA 参数需要
  显式包成 replicated DTensor，避免优化器和 checkpoint 语义不一致。
- 对 `fastvideo/train/` 来说，把 LoRA 作为 `models.student` 的可选构造参数
  最薄，也最不破坏当前配置结构。
- `Wan-AI/Wan2.2-TI2V-5B-Diffusers` 和 `Wan-AI/Wan2.1-I2V-*` 不能混为一谈。
  前者在仓库现成 distill 脚本里继续使用 T2V 风格 parquet 预处理；后者才需要
  `clip_feature` / `first_frame_latent` 这类 I2V parquet 字段。
- 旧的 `fastvideo/training/wan_i2v_training_pipeline.py` 适合迁移给 Wan I2V
  family，但不应该直接套到 Wan2.2 TI2V 5B 上，否则会把 TI2V 错误地绑到
  `image_encoder` 依赖上。
- `MatrixGame` 在新训练栈里不是简单的 `Wan I2V` 变体。它除了
  `clip_feature` / `first_frame_latent` 之外，还要求
  `keyboard_cond` / `mouse_cond` 进入 transformer forward；因此最薄的迁移
  方式是新增一个 `MatrixGameModel(WanModel)`，复用 Wan 的噪声调度和 LoRA
  注入，但覆写 batch 组装和 validation action 透传。
- `GEN3C-Cosmos-7B` 当前代码层面只完成了推理路径：`Gen3CPipeline` 依赖
  `image_path`、MoGe 深度估计、3D cache 渲染、`condition_video_pose` /
  `condition_video_input_mask` 这套专用 conditioning。虽然 transformer 内部
  attention 命名（`to_q/to_k/to_v/to_out`）与训练侧 LoRA 匹配规则兼容，但
  新训练栈缺少对应的 model plugin、parquet schema、preprocess/validation
  数据链和训练 batch 组装，因此不能像 Wan/Hunyuan 那样只换 `init_from`
  就跑通。

## Mistakes / Dead Ends
- 直接复用 `LoRAPipeline` 本身并不合适，因为它绑定了完整 pipeline 生命周期，
  对新框架的 model plugin 来说过重。
- 当前本地环境缺少完整测试依赖，只能做 `py_compile` 级别校验，无法在本地跑
  真实训练或 pytest。

## Proposed Standardization
- 如果后续需要推广到更多 backbone，可以把 `fastvideo/train/utils/lora.py`
  提升为通用训练侧 LoRA 接口，并逐个补齐对应 model plugin（例如 Wan I2V、
  MatrixGame、LongCat、GEN3C）。
- 若需要产出可直接推理的 adapter，建议补一个 `fastvideo/train` 下的
  DCP-to-LoRA 导出入口，而不是只依赖全量 diffusers 导出。
