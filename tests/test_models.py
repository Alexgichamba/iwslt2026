"""Smoke tests for model construction and forward passes."""

import torch
import pytest


def test_speech_encoder_forward():
    from st.models.speech_encoder import SpeechEncoder

    model = SpeechEncoder(
        input_dim=80, encoder_dim=64, num_heads=4, ffn_dim=128,
        num_layers=2, depthwise_conv_kernel_size=7, vocab_size=32,
    )

    B, T, F = 2, 100, 80
    features = torch.randn(B, T, F)
    lengths = torch.tensor([100, 80])

    out = model(features, lengths)
    assert "hidden_states" in out
    assert "ctc_logits" in out
    assert out["hidden_states"].shape[0] == B
    assert out["hidden_states"].shape[2] == 64
    assert out["ctc_logits"].shape[2] == 32


def test_speech_encoder_no_ctc():
    from st.models.speech_encoder import SpeechEncoder

    model = SpeechEncoder(
        input_dim=80, encoder_dim=64, num_heads=4, ffn_dim=128,
        num_layers=2, depthwise_conv_kernel_size=7, vocab_size=None,
    )

    out = model(torch.randn(1, 100, 80), torch.tensor([100]))
    assert "ctc_logits" not in out


def test_projectors():
    from st.models.projector import build_projector

    B, T, D_enc, D_llm = 2, 50, 64, 128
    x = torch.randn(B, T, D_enc)
    lengths = torch.tensor([50, 40])

    for name in ["linear", "mlp", "conv", "stack"]:
        proj = build_projector(name, D_enc, D_llm)
        out, out_lengths = proj(x, lengths)
        assert out.shape[0] == B
        assert out.shape[2] == D_llm

        if name in ("conv", "stack"):
            assert out.shape[1] < T  # should downsample


def test_projector_registry_error():
    from st.models.projector import build_projector

    with pytest.raises(ValueError, match="Unknown projector"):
        build_projector("nonexistent", 64, 128)


def test_encoder_freeze_unfreeze():
    from st.models.speech_encoder import SpeechEncoder

    model = SpeechEncoder(
        input_dim=80, encoder_dim=64, num_heads=4, ffn_dim=128,
        num_layers=2, depthwise_conv_kernel_size=7,
    )

    model.freeze()
    assert all(not p.requires_grad for p in model.parameters())

    model.unfreeze()
    assert all(p.requires_grad for p in model.parameters())
