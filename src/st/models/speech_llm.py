"""
End-to-end Speech LLM: encoder → [CTC compress] → projector → LLM.

This is the main model class for speech translation. It handles:
- Encoding audio through the speech encoder
- Optionally compressing encoder hidden states via CTC predictions
- Projecting to LLM embedding space via a chosen projector
- Constructing combined (speech prefix + text) input embeddings
- Computing cross-entropy loss for translation targets
- Generation at inference time

Training stages:
    Stage 1 (CTC pretrain): Use SpeechEncoder directly — this module not needed.
    Stage 2 (Projector training): encoder.freeze(), llm frozen, train projector.
    Stage 3 (End-to-end LoRA): encoder.freeze(), projector trainable, LoRA active on LLM.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from st.models.speech_encoder import SpeechEncoder
from st.models.projector import build_projector
from st.models.ctc_compressor import CTCCompressorV2, build_ctc_compressor
from st.models.llm_wrapper import LLMWrapper


class SpeechLLM(nn.Module):
    """Full speech-to-text translation model.

    Architecture:
        encoder → [ctc_compressor] → projector → LLM

    The CTC compressor is optional. When present, it uses the encoder's CTC
    head to adaptively compress hidden states before the projector. This
    requires the encoder to have a CTC head (vocab_size != None).

    Args:
        encoder: Pretrained speech encoder.
        projector_name: Projector type ('mlp', 'concat', 'conv1d', 'transformer', 'qformer').
        llm: LLM wrapper instance.
        projector_kwargs: Extra kwargs for projector constructor.
        ctc_compressor: Optional CTC compressor module.
    """

    def __init__(
        self,
        encoder: SpeechEncoder,
        projector_name: str,
        llm: LLMWrapper,
        projector_kwargs: dict | None = None,
        ctc_compressor: CTCCompressorV2 | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.ctc_compressor = ctc_compressor

        if ctc_compressor is not None and encoder.ctc_head is None:
            raise ValueError(
                "CTC compressor requires encoder to have a CTC head (vocab_size != None). "
                "Load the encoder checkpoint with vocab_size set."
            )

        projector_kwargs = projector_kwargs or {}
        self.projector = build_projector(
            name=projector_name,
            encoder_dim=encoder.get_output_dim(),
            llm_dim=llm.get_embedding_dim(),
            **projector_kwargs,
        )

    def _encode_and_compress(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run encoder and optional CTC compression.

        Returns:
            hidden: (B, T', D) — encoder hidden states, possibly compressed
            lengths: (B,) — sequence lengths after compression
        """
        enc_out = self.encoder(features, feature_lengths)
        hidden = enc_out["hidden_states"]
        lengths = enc_out["lengths"]

        if self.ctc_compressor is not None:
            ctc_logits = enc_out["ctc_logits"]  # requires CTC head
            hidden, lengths = self.ctc_compressor(hidden, ctc_logits, lengths)

        return hidden, lengths

    def _project(
        self,
        hidden: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project encoder hidden states to LLM space with dtype casting."""
        projected, proj_lengths = self.projector(hidden, lengths)
        # Cast to LLM dtype (projector is float32, LLM may be bfloat16)
        llm_dtype = self.llm.get_input_embeddings().weight.dtype
        return projected.to(llm_dtype), proj_lengths

    def _build_inputs(
        self,
        speech_embeds: torch.Tensor,
        speech_lengths: torch.Tensor,
        text_input_ids: torch.Tensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Construct combined [speech_embeds | text_embeds] inputs for the LLM.

        Returns:
            inputs_embeds: (B, T_speech + T_text, D_llm)
            attention_mask: (B, T_speech + T_text)
            speech_seq_len: length of speech prefix (for label masking)
        """
        b = speech_embeds.size(0)
        device = speech_embeds.device
        max_speech_len = speech_embeds.size(1)

        speech_mask = torch.arange(max_speech_len, device=device).unsqueeze(0) < speech_lengths.unsqueeze(1)

        if text_input_ids is not None:
            text_embeds = self.llm.get_input_embeddings()(text_input_ids)
            inputs_embeds = torch.cat([speech_embeds, text_embeds], dim=1)

            if text_attention_mask is None:
                text_attention_mask = torch.ones(
                    text_input_ids.shape, dtype=torch.bool, device=device
                )
            attention_mask = torch.cat(
                [speech_mask.to(text_attention_mask.dtype), text_attention_mask], dim=1
            )
        else:
            inputs_embeds = speech_embeds
            attention_mask = speech_mask.long()

        return inputs_embeds, attention_mask, max_speech_len

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        text_input_ids: torch.Tensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            features: (B, T, F) log-mel spectrogram
            feature_lengths: (B,) frame counts
            text_input_ids: (B, T_text) tokenized translation target
            text_attention_mask: (B, T_text) mask for text
            labels: (B, T_text) target IDs for loss computation

        Returns:
            dict with "loss" and/or "logits"
        """
        hidden, enc_lengths = self._encode_and_compress(features, feature_lengths)
        projected, proj_lengths = self._project(hidden, enc_lengths)

        inputs_embeds, attention_mask, speech_len = self._build_inputs(
            projected, proj_lengths, text_input_ids, text_attention_mask
        )

        if labels is not None:
            ignore = torch.full((labels.size(0), speech_len), -100, device=labels.device)
            full_labels = torch.cat([ignore, labels], dim=1)
        else:
            full_labels = None

        output = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=full_labels,
        )

        result = {"logits": output.logits}
        if output.loss is not None:
            result["loss"] = output.loss

        return result

    @torch.no_grad()
    def translate(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        **generate_kwargs,
    ) -> list[str]:
        """End-to-end speech → text translation.

        Args:
            features: (B, T, F) log-mel spectrogram
            feature_lengths: (B,) frame counts
            **generate_kwargs: e.g. max_new_tokens=256, num_beams=4

        Returns:
            List of translated strings.
        """
        hidden, enc_lengths = self._encode_and_compress(features, feature_lengths)
        projected, proj_lengths = self._project(hidden, enc_lengths)

        inputs_embeds, attention_mask, _ = self._build_inputs(projected, proj_lengths)

        generate_kwargs.setdefault("max_new_tokens", 256)

        token_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return self.llm.tokenizer.batch_decode(token_ids, skip_special_tokens=True)