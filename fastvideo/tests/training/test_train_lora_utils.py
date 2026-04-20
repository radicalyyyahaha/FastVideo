# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch.nn as nn

from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.lora.linear import BaseLayerWithLoRA
from fastvideo.train.utils.lora import enable_lora_training


class _ArchConfig:
    exclude_lora_layers = ["embedder"]


class _Config:
    arch_config = _ArchConfig()


class _ToyBlock(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = ReplicatedLinear(8, 8, bias=False)
        self.k_proj = ReplicatedLinear(8, 8, bias=False)
        self.ff = ReplicatedLinear(8, 8, bias=False)


class _ToyTransformer(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.blocks = nn.ModuleList([_ToyBlock()])
        self.embedder = ReplicatedLinear(8, 8, bias=False)


def test_enable_lora_training_replaces_only_targeted_layers() -> None:
    transformer = _ToyTransformer()

    converted = enable_lora_training(
        transformer,
        lora_rank=4,
        lora_alpha=8,
        lora_target_modules=["q_proj", "k_proj", "embedder"],
    )

    assert converted == 2
    assert isinstance(transformer.blocks[0].q_proj, BaseLayerWithLoRA)
    assert isinstance(transformer.blocks[0].k_proj, BaseLayerWithLoRA)
    assert not isinstance(transformer.blocks[0].ff, BaseLayerWithLoRA)
    assert not isinstance(transformer.embedder, BaseLayerWithLoRA)

    trainable_names = {
        name
        for name, param in transformer.named_parameters()
        if param.requires_grad
    }
    assert trainable_names == {
        "blocks.0.q_proj.lora_A",
        "blocks.0.q_proj.lora_B",
        "blocks.0.k_proj.lora_A",
        "blocks.0.k_proj.lora_B",
    }
