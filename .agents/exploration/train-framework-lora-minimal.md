# Exploration Log: Train Framework LoRA Minimal

## Status: under_review

## Context

`fastvideo/train/` is the preferred YAML-driven training framework. Before this
work, practical LoRA training support mostly lived in the legacy
`fastvideo/training/` pipelines. The goal was to add a minimal LoRA finetuning
path to the new framework by injecting LoRA layers inside model plugins, rather
than copying entire legacy training pipelines.

## Progress

- [x] Read the new training framework, inference-side LoRA layers, and legacy
  training pipelines.
- [x] Identified the missing piece in the new framework: no training-side LoRA
  injection hook.
- [x] Added minimal injection logic in `fastvideo/train/utils/lora.py`.
- [x] Wired LoRA parameters into `WanModel`, `HunyuanModel`, and
  `WanCausalModel`.
- [x] Added minimal YAML examples and documentation.
- [x] Extended the new `WanModel` training path from pure T2V batches to
  I2V/TI2V-compatible parquet where appropriate.
- [x] Added a minimal LoRA YAML example and lightweight tests for
  `Wan2.2 TI2V 5B`.
- [x] Added a `Matrix-Game-2.0` model plugin, action-conditioned validation
  passthrough, and a LoRA YAML example for the new training stack.
- [x] Reviewed `GEN3C-Cosmos-7B` and confirmed that the current repository only
  provides inference integration for it. It is not wired into the
  `fastvideo/train/` LoRA finetuning path.

## Findings

- The inference-side `LoRAPipeline` already has reusable LoRA layer wrappers.
  The missing training-side concept was when to replace transformer layers with
  LoRA-wrapped layers.
- The new framework loads and shards the transformer before the model plugin
  returns it. Newly added LoRA parameters therefore need to be explicit
  replicated DTensors so optimizer and checkpoint semantics remain consistent.
- Adding LoRA as optional `models.student` constructor arguments is the thinnest
  integration point for `fastvideo/train/`.
- `Wan-AI/Wan2.2-TI2V-5B-Diffusers` should not be treated the same as
  `Wan-AI/Wan2.1-I2V-*`. The former uses T2V-style parquet in the available
  training path. The latter needs I2V fields such as `clip_feature` and
  `first_frame_latent`.
- The legacy `fastvideo/training/wan_i2v_training_pipeline.py` is useful
  reference material for Wan I2V models, but it should not be applied directly
  to Wan2.2 TI2V 5B because that incorrectly introduces an `image_encoder`
  dependency.
- `MatrixGame` is not just a plain Wan I2V variant. In addition to
  `clip_feature` and `first_frame_latent`, it needs `keyboard_cond` and
  `mouse_cond` in the transformer forward path. The thinnest migration is a
  `MatrixGameModel(WanModel)` that reuses Wan scheduling and LoRA injection but
  overrides batch preparation and validation action passthrough.
- `GEN3C-Cosmos-7B` currently has inference-only plumbing. `Gen3CPipeline`
  depends on `image_path`, MoGe depth estimation, 3D cache rendering,
  `condition_video_pose`, and `condition_video_input_mask`. Its transformer
  attention names (`to_q/to_k/to_v/to_out`) are compatible with the generic LoRA
  target matcher, but the new training stack lacks the required model plugin,
  parquet schema, preprocessing path, validation data path, and training batch
  assembly.

## Mistakes / Dead Ends

- Reusing `LoRAPipeline` directly was too heavy. It is tied to full pipeline
  lifecycle, while the new training stack needs a model-plugin-level hook.
- The local environment did not include all test dependencies, so local
  validation was limited to `py_compile` and YAML parsing rather than full
  pytest or real training launches.

## Proposed Standardization

- Promote `fastvideo/train/utils/lora.py` into the shared training-side LoRA
  integration point as more backbones are added.
- For future families, add one focused model plugin per distinct conditioning
  contract rather than overloading `WanModel`.
- Add a `fastvideo/train` export path from DCP checkpoints to standalone LoRA
  adapters, instead of relying only on full Diffusers export.
