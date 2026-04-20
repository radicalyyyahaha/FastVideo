# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from fastvideo.train.models.matrixgame.matrixgame import (
    MatrixGameModel,
)


def _make_model() -> MatrixGameModel:
    model = object.__new__(MatrixGameModel)
    model._init_from = "FastVideo/Matrix-Game-2.0-Base-Diffusers"
    model.training_config = SimpleNamespace(
        model_path=model._init_from,
        data=SimpleNamespace(
            num_latent_t=20,
            num_frames=77,
        ),
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(
                    temporal_compression_ratio=4,
                ),
            ),
        ),
    )
    return model


def test_matrixgame_always_uses_image_conditioning() -> None:
    model = _make_model()
    assert model._uses_image_conditioning() is True


def test_matrixgame_build_distill_kwargs_includes_action_conditioning() -> None:
    model = _make_model()
    noise_input = torch.randn(1, 20, 36, 4, 4)
    timestep = torch.tensor([42])
    mouse_cond = torch.randn(1, 77, 2)
    keyboard_cond = torch.randn(1, 77, 6)

    kwargs = model._build_distill_input_kwargs(
        noise_input,
        timestep,
        {
            "encoder_hidden_states": None,
            "encoder_attention_mask": None,
            "encoder_hidden_states_image": torch.randn(1, 257, 1280),
            "mouse_cond": mouse_cond,
            "keyboard_cond": keyboard_cond,
        },
    )

    assert kwargs["hidden_states"].shape == (1, 36, 20, 4, 4)
    assert kwargs["mouse_cond"] is mouse_cond
    assert kwargs["keyboard_cond"] is keyboard_cond


def test_matrixgame_uncond_dict_preserves_action_conditioning() -> None:
    model = _make_model()
    mouse_cond = torch.randn(2, 77, 2)
    keyboard_cond = torch.randn(2, 77, 6)

    def _fake_ensure() -> None:
        model.negative_prompt_embeds = torch.randn(1, 8, 16)
        model.negative_prompt_attention_mask = torch.ones(1, 8)

    model.ensure_negative_conditioning = _fake_ensure  # type: ignore[method-assign]

    batch = SimpleNamespace(
        mouse_cond=mouse_cond,
        keyboard_cond=keyboard_cond,
        conditional_dict={"encoder_hidden_states_image": torch.randn(2, 257, 1280)},
        unconditional_dict=None,
        encoder_hidden_states=None,
    )

    out = model._get_uncond_text_dict(batch, cfg_uncond=None)

    assert out["mouse_cond"] is mouse_cond
    assert out["keyboard_cond"] is keyboard_cond
