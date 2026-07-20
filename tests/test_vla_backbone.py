from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from torch import nn

from evo_rlt.adapters.lerobot.policies.vla_backbone import (
    _pi05_checkpoint_is_device_direct_compatible,
    extract_prefix_hidden,
    get_vla_prefix_target,
    infer_num_image_tokens,
    infer_vla_token_dim,
    infer_vla_type,
)


def test_infer_vla_type_from_config_json(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"type": "smolvla"}))

    assert infer_vla_type(str(tmp_path)) == "smolvla"


def test_pi05_current_format_checkpoint_can_use_device_direct_load(tmp_path):
    from safetensors.torch import save_file

    model_file = tmp_path / "model.safetensors"
    save_file({"model.layer.weight": torch.zeros(1)}, model_file)

    assert _pi05_checkpoint_is_device_direct_compatible(model_file)


def test_pi05_legacy_format_checkpoint_uses_fallback_loader(tmp_path):
    from safetensors.torch import save_file

    model_file = tmp_path / "model.safetensors"
    save_file({"layer.weight": torch.zeros(1)}, model_file)

    assert not _pi05_checkpoint_is_device_direct_compatible(model_file)


def test_get_smolvla_prefix_target_and_extract_hidden():
    target = object()
    policy = SimpleNamespace(model=SimpleNamespace(vlm_with_expert=target))
    prefix = torch.randn(2, 3, 4)

    assert get_vla_prefix_target(policy) is target
    assert extract_prefix_hidden(((prefix, None), "past")) is prefix


def test_infer_token_dim_from_smolvla_shape():
    policy = SimpleNamespace(
        model=SimpleNamespace(
            vlm_with_expert=SimpleNamespace(
                config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=576))
            )
        )
    )

    assert infer_vla_token_dim(policy) == 576


def test_infer_token_dim_from_linear_test_double():
    assert infer_vla_token_dim(nn.Linear(8, 16)) == 16


def test_infer_smolvla_image_tokens_from_vision_config():
    policy = SimpleNamespace(
        config=SimpleNamespace(
            image_features=["cam0", "cam1"],
            resize_imgs_with_padding=(512, 512),
            add_image_special_tokens=False,
        ),
        model=SimpleNamespace(
            vlm_with_expert=SimpleNamespace(
                config=SimpleNamespace(vision_config=SimpleNamespace(patch_size=16))
            )
        ),
    )

    assert infer_num_image_tokens(policy, required=True) == 2 * 32 * 32
