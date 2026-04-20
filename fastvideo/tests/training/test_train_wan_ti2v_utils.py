# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from fastvideo.train.models.wan.wan import WanModel


def _make_wan_model(*, model_path: str, ti2v_task: bool) -> WanModel:
    model = object.__new__(WanModel)
    model._init_from = model_path
    model.training_config = SimpleNamespace(
        model_path=model_path,
        data=SimpleNamespace(num_latent_t=20),
        pipeline_config=SimpleNamespace(
            ti2v_task=ti2v_task,
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(temporal_compression_ratio=4),
            ),
        ),
    )
    return model


def test_wan_image_conditioning_detection_matches_model_id() -> None:
    ti2v_model = _make_wan_model(
        model_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        ti2v_task=True,
    )
    i2v_model = _make_wan_model(
        model_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        ti2v_task=False,
    )

    assert ti2v_model._uses_image_conditioning() is False
    assert i2v_model._uses_image_conditioning() is True


def test_wan_ti2v_mask_and_image_latents_are_concatenated() -> None:
    model = _make_wan_model(
        model_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        ti2v_task=True,
    )
    noisy_model_input = torch.randn(2, 16, 20, 4, 4)
    image_latents = torch.randn(2, 16, 20, 4, 4)

    augmented, mask_lat_size = model._augment_noisy_input_with_image_condition(
        noisy_model_input,
        image_latents,
    )

    assert mask_lat_size.shape == (2, 4, 20, 4, 4)
    assert torch.all(mask_lat_size[:, :, :1] == 1)
    assert torch.all(mask_lat_size[:, :, 1:] == 0)
    assert augmented.shape == (2, 36, 20, 4, 4)
    assert torch.equal(augmented[:, :16], noisy_model_input)
    assert torch.equal(augmented[:, 20:], image_latents)


def test_build_distill_kwargs_includes_image_conditioning() -> None:
    model = _make_wan_model(
        model_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        ti2v_task=True,
    )
    noise_input = torch.randn(1, 20, 36, 4, 4)
    timestep = torch.tensor([42])
    image_embeds = torch.randn(1, 257, 1280)

    kwargs = model._build_distill_input_kwargs(
        noise_input,
        timestep,
        {
            "encoder_hidden_states": torch.randn(1, 512, 4096),
            "encoder_attention_mask": torch.ones(1, 512),
            "encoder_hidden_states_image": image_embeds,
        },
    )

    assert kwargs["hidden_states"].shape == (1, 36, 20, 4, 4)
    assert kwargs["encoder_hidden_states_image"] is image_embeds


def test_negative_prompt_conditioning_is_built_lazily() -> None:
    model = _make_wan_model(
        model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        ti2v_task=False,
    )

    called = {"count": 0}

    def _fake_ensure() -> None:
        called["count"] += 1
        model.negative_prompt_embeds = torch.randn(1, 8, 16)
        model.negative_prompt_attention_mask = torch.ones(1, 8)

    model.ensure_negative_conditioning = _fake_ensure  # type: ignore[method-assign]
    batch = SimpleNamespace(
        encoder_hidden_states=torch.randn(2, 8, 16),
        conditional_dict=None,
        unconditional_dict=None,
    )

    out = model._get_uncond_text_dict(batch, cfg_uncond=None)

    assert called["count"] == 1
    assert out["encoder_hidden_states"].shape == (2, 8, 16)
    assert out["encoder_attention_mask"].shape == (2, 8)
