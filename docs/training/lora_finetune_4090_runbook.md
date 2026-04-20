# FastVideo LoRA Finetune Migration and RTX 4090 Runbook

This runbook documents the work done to add minimal LoRA finetuning support to
the YAML-driven `fastvideo/train/` framework, then use that path to test common
models from the inference support matrix on a single RTX 4090. The goal was not
to produce full production training recipes. The goal was to answer practical
questions:

- Which model families can launch LoRA finetuning in the current framework?
- Which ones fail because of unsupported data or model plumbing?
- Which ones are simply too large for a 24 GB 4090?
- Which data and preprocessing paths are required?
- How can each run be recorded into an Excel tracker?

## Background

FastVideo already had inference-side LoRA utilities and several legacy training
pipelines. The newer modular training stack under `fastvideo/train/` did not
have a thin, reusable LoRA injection path. The implementation added here keeps
the change small:

- Model plugins load the base transformer as before.
- LoRA is injected immediately after transformer loading.
- The training method remains `FineTuneMethod`.
- YAML controls `lora_rank`, `lora_alpha`, and `lora_target_modules`.
- `examples/train/run.sh` creates a structured run record after every launch
  and optionally appends it to an Excel workbook.

This is a debug and bring-up path. It is useful for feasibility testing and
small experiments, but it should not be confused with an official full-scale
training recipe for every model in the inference matrix.

## Main Changes

### Training-side LoRA Injection

Core files:

- `fastvideo/train/utils/lora.py`
- `fastvideo/train/models/wan/wan.py`
- `fastvideo/train/models/hunyuan/hunyuan.py`
- `fastvideo/train/models/wan/wan_causal.py`

`enable_lora_training(...)` does the following:

- Freezes the base transformer.
- Finds compatible linear projection layers by module-name pattern.
- Replaces those layers with `BaseLayerWithLoRA`.
- Wraps newly added LoRA parameters as replicated DTensors when distributed
  training has already initialized model sharding.

Default target patterns:

```yaml
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - to_q
  - to_k
  - to_v
  - to_out
  - to_qkv
  - to_gate_compress
```

For Wan-family models, prefer explicit targets:

```yaml
lora_target_modules:
  - to_q
  - to_k
  - to_v
  - to_out
```

For Hunyuan-family models, use:

```yaml
lora_target_modules:
  - img_attn_qkv
  - img_attn_proj
  - txt_attn_qkv
  - txt_attn_proj
  - self_attn_qkv
  - self_attn_proj
```

### Excel Run Logging

Core files:

- `examples/train/run.sh`
- `scripts/training/build_run_record.py`
- `scripts/training/append_run_log.py`

Example launch:

```bash
TRAINING_RUN_LOG_XLSX="/workspace/FastVideo/result/train_log.xlsx" \
TRAINING_RUN_LOG_SHEET="train_log" \
TRAINING_RUN_LOG_OWNER="$USER" \
NUM_GPUS=1 WANDB_MODE=offline \
bash examples/train/run.sh \
  examples/train/configs/fine_tuning/wan/t2v_lora.yaml
```

The launcher writes a run record for both successful and failed runs. On
failure, it records `status` and `failure_reason`. On success, it attempts to
record:

- `final_train_loss`
- `avg_step_time_sec`
- `peak_vram_gb`
- `wall_time_hours`
- model ID, data path, resolution, LoRA rank, precision, and output directory

`peak_vram_gb` currently comes from `torch.cuda.max_memory_allocated()`. This is
the PyTorch tensor allocation peak. It is not the same as total process VRAM in
`nvidia-smi`, which also includes allocator reservations, CUDA context, NCCL,
cuDNN, FlashAttention workspaces, and other non-PyTorch allocations.

## Data Preparation

### Wan T2V Data

Example dataset:

```bash
python scripts/huggingface/download_hf.py \
  --repo_id "wlsaidhi/crush-smol-merged" \
  --local_dir "data/crush-smol" \
  --repo_type "dataset"
```

Create `merge.txt`:

```bash
printf 'data/crush-smol/videos,data/crush-smol/videos2caption.json\n' \
  > data/crush-smol/merge.txt
```

The older preprocessing entrypoint was the most reliable path in this
environment:

```bash
torchrun --nproc_per_node=1 \
  fastvideo/pipelines/preprocess/v1_preprocess.py \
  --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --data_merge_path data/crush-smol/merge.txt \
  --preprocess_video_batch_size 1 \
  --seed 42 \
  --max_height 480 \
  --max_width 832 \
  --num_frames 77 \
  --dataloader_num_workers 0 \
  --output_dir data/crush-smol_processed_t2v \
  --train_fps 16 \
  --samples_per_file 4 \
  --flush_frequency 4 \
  --video_length_tolerance_range 5 \
  --preprocess_task t2v
```

Training YAML should point to:

```yaml
training:
  data:
    data_path: data/crush-smol_processed_t2v/combined_parquet_dataset
```

### Wan2.2 TI2V 5B Data

In the current new training stack, `Wan-AI/Wan2.2-TI2V-5B-Diffusers` still uses
T2V-style parquet data. It should not be treated like Wan I2V parquet with
`clip_feature` and `first_frame_latent`.

```bash
torchrun --nproc_per_node=1 \
  fastvideo/pipelines/preprocess/v1_preprocess.py \
  --model_path Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --data_merge_path data/crush-smol/merge.txt \
  --preprocess_video_batch_size 1 \
  --seed 42 \
  --max_height 480 \
  --max_width 832 \
  --num_frames 77 \
  --dataloader_num_workers 0 \
  --output_dir data/crush-smol_processed_ti2v \
  --train_fps 16 \
  --samples_per_file 1 \
  --flush_frequency 1 \
  --video_length_tolerance_range 5 \
  --preprocess_task t2v
```

Training YAML should point to:

```yaml
training:
  data:
    data_path: data/crush-smol_processed_ti2v/combined_parquet_dataset
```

### MatrixGame Data

MatrixGame is not a plain Wan I2V variant. Its real training schema requires:

- `clip_feature`
- `first_frame_latent`
- `keyboard_cond`
- `mouse_cond`

For pure bring-up, empty or synthetic action tensors can be used to test the
training path. That does not produce a meaningful action-conditioned finetune.

The repository example assumes a local `footsies-dataset/` directory:

```bash
python fastvideo/pipelines/preprocess/v1_preprocess.py \
  --model_path FastVideo/Matrix-Game-2.0-Foundation-Diffusers \
  --data_merge_path footsies-dataset/merge.txt \
  --preprocess_video_batch_size 4 \
  --seed 42 \
  --max_height 352 \
  --max_width 640 \
  --num_frames 77 \
  --dataloader_num_workers 0 \
  --output_dir footsies-dataset/preprocessed \
  --samples_per_file 4 \
  --train_fps 25 \
  --flush_frequency 4 \
  --preprocess_task matrixgame
```

If a real I2V parquet dataset already exists and includes `clip_feature` plus
`first_frame_latent`, it can be used temporarily to debug MatrixGame training.
T2V/TI2V parquet cannot be used directly.

### Hunyuan and FastHunyuan Data

The Hunyuan debug YAML defaults to:

```text
data/hunyuan_overfit_preprocessed
```

That directory is not automatically downloaded. It is expected to be generated
from local raw videos under `data/hunyuan_overfit/`. `FastVideo/FastHunyuan-
diffusers` should be treated as a Hunyuan 13B-class model. A single RTX 4090 is
very likely to OOM for LoRA training.

### GEN3C Data

`FastVideo/GEN3C-Cosmos-7B-Diffusers` currently has an inference path, not a
new-framework LoRA training path.

It is not normal I2V data. Inference requires:

- `image_path`
- MoGe depth estimation
- 3D cache rendering
- `condition_video_pose`
- `condition_video_input_mask`
- camera trajectory controls: `trajectory_type`, `movement_distance`,
  `camera_rotation`

Existing T2V/I2V parquet data cannot be reused directly for GEN3C training.

## Model Results

| Model | YAML / Path | RTX 4090 Result | Notes |
| --- | --- | --- | --- |
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `examples/train/configs/fine_tuning/wan/t2v_lora.yaml` | Runs | Most stable baseline |
| `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` | `examples/train/configs/fine_tuning/wan/fast_t2v_lora_vsa.yaml` | Tryable | Requires VSA; selected by `training.vsa.sparsity` |
| `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | `examples/train/configs/fine_tuning/wan/ti2v_lora.yaml` | Tryable with bf16 | Very close to the 24 GB limit |
| `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` | `examples/train/configs/fine_tuning/wan/fast_ti2v_fullattn_lora_vsa.yaml` | Tryable, high risk | Uses VSA; still uses T2V-style parquet |
| `loayrashid/TurboWan2.1-T2V-1.3B-Diffusers` | `examples/train/configs/fine_tuning/wan/turbo_t2v_lora_sla.yaml` | Tryable | Requires SLA attention |
| `hunyuanvideo-community/HunyuanVideo` | `examples/train/configs/fine_tuning/hunyuan/t2v_lora.yaml` | Likely OOM | Hunyuan 13B-class model |
| `FastVideo/FastHunyuan-diffusers` | `examples/train/configs/fine_tuning/hunyuan/fast_t2v_lora.yaml` | Likely OOM | bf16 weights alone are still about 26 GB-class |
| `FastVideo/Matrix-Game-2.0-*` | `examples/train/configs/fine_tuning/matrixgame/i2v_lora.yaml` | Data-dependent | Needs MatrixGame schema for meaningful training |
| `FastVideo/GEN3C-Cosmos-7B-Diffusers` | No training YAML | Unsupported | Inference support only in the current framework |

## Launch Command

Generic launch:

```bash
cd /workspace/FastVideo
export PYTHONPATH=/workspace/FastVideo:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAINING_RUN_LOG_XLSX="/workspace/FastVideo/result/train_log.xlsx" \
TRAINING_RUN_LOG_SHEET="train_log" \
TRAINING_RUN_LOG_OWNER="$USER" \
NUM_GPUS=1 WANDB_MODE=offline \
bash examples/train/run.sh \
  examples/train/configs/fine_tuning/wan/t2v_lora.yaml
```

MatrixGame variants can be selected with overrides:

```bash
bash examples/train/run.sh \
  examples/train/configs/fine_tuning/matrixgame/i2v_lora.yaml \
  --models.student.init_from FastVideo/Matrix-Game-2.0-GTA-Diffusers \
  --training.checkpoint.output_dir outputs/matrixgame_gta_i2v_lora_debug \
  --training.tracker.run_name matrixgame_gta_i2v_lora_debug
```

## Issue Log

### `ModuleNotFoundError: No module named 'fastvideo'`

The training process did not have the repository root on `PYTHONPATH`.
`run.sh` now sets:

```bash
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python -m torch.distributed.run -m fastvideo.train.entrypoint.train ...
```

Manual launches should also set:

```bash
export PYTHONPATH=/workspace/FastVideo:$PYTHONPATH
```

### `No parquet files found`

`training.data.data_path` must point to a preprocessed
`combined_parquet_dataset` directory. Raw video directories cannot be passed
directly to the new training stack.

### `v1_preprocessing_new.py: unrecognized arguments`

Some environments include `v1_preprocessing_new.py` without the `--preprocess.*`
CLI surface. Use the older entrypoint instead:

```bash
fastvideo/pipelines/preprocess/v1_preprocess.py
```

### Preprocessing OOM or High System Memory

Reduce these first:

- `--preprocess_video_batch_size 1`
- `--samples_per_file 1`
- `--flush_frequency 1`
- resolution
- frame count

On RunPod, free disk space in this order:

- `outputs/`
- training logs
- generated videos
- unused preprocessing outputs
- unused large Hugging Face model caches

### VAE or Text Encoder dtype/device mismatch

When preprocessing uses bf16 or CPU offload, the input tensor and the module
weights must be on the same device and use compatible dtype. Typical failures:

- `Input type (float) and bias type (BFloat16) should be the same`
- `Input type (CUDABFloat16Type) and weight type (CPUBFloat16Type) should be the same`

### `Inference tensors do not track version counter`

FSDP forward hooks do not work with tensors created under
`torch.inference_mode()`. Use `torch.no_grad()` for preprocessing and encoding
paths that still pass through FSDP-managed modules.

### TI2V validation pipeline

For `Wan-AI/Wan2.2-TI2V-5B-Diffusers` and the FastWan TI2V path, validation in
the new training stack should use:

```yaml
pipeline_target: fastvideo.pipelines.basic.wan.wan_pipeline.WanPipeline
```

Do not switch to `WanImageToVideoPipeline`; that pipeline expects an
`image_encoder` module that these checkpoints do not expose.

### VSA Logs Also Mention FlashAttention

This is expected. When Wan uses VSA, video self-attention can use
`VIDEO_SPARSE_ATTN`, while text/image cross-attention can still fall back to
FlashAttention. Seeing both backends in the log does not mean VSA failed.

### 5B Model Uses Almost All VRAM Before Training

With fp32 master weights, a 5B model needs roughly 20 GB for parameters alone.
After CUDA context, buffers, attention workspaces, and validation components,
an RTX 4090 has almost no headroom. Use:

```yaml
training:
  dit_precision: bf16
```

Also use:

- `enable_gradient_checkpointing_type: full`
- `train_batch_size: 1`
- disabled or infrequent validation when debugging
- smaller `num_latent_t` when necessary

### `avg_step_time_sec` Is Not a Full Run Time

`avg_step_time_sec` only measures training step time. It excludes model loading,
preprocessing, and validation. Major factors include:

- `num_latent_t`
- precision: `fp32` vs `bf16`
- `gradient_accumulation_steps`
- model size
- attention backend
- LoRA rank

Do not compare two runs directly if `num_latent_t`, precision, or validation
settings differ.

## Minimal Migration File List

For "LoRA training plus Excel logging", the important files are:

- `fastvideo/train/utils/lora.py`
- `fastvideo/train/models/wan/wan.py`
- `fastvideo/train/models/hunyuan/hunyuan.py`
- `fastvideo/train/models/matrixgame/matrixgame.py`
- `fastvideo/train/utils/dataloader.py`
- `fastvideo/pipelines/pipeline_batch_info.py`
- `fastvideo/train/callbacks/validation.py`
- `fastvideo/train/entrypoint/train.py`
- `fastvideo/models/loader/fsdp_load.py`
- `examples/train/run.sh`
- `scripts/training/build_run_record.py`
- `scripts/training/append_run_log.py`
- `examples/train/configs/fine_tuning/**/**_lora*.yaml`

Docs and tests are not required at runtime, but they should be kept when
possible:

- `docs/training/finetune.md`
- `docs/training/lora_finetune_4090_runbook.md`
- `fastvideo/tests/training/test_train_*_utils.py`

## Follow-up Work

- Add a DCP checkpoint to standalone LoRA adapter export path under
  `fastvideo/train`.
- Record both allocated and reserved VRAM peaks to avoid confusion with
  `nvidia-smi`.
- Design a dedicated GEN3C training schema. Prefer offline 3D cache
  conditioning to avoid running MoGe inside the training step.
- Use real keyboard and mouse action data for meaningful MatrixGame finetuning.
