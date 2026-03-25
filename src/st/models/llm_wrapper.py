"""
Thin wrapper around HuggingFace causal language models.

Responsibilities:
- Load any AutoModelForCausalLM by name/path
- Optionally apply LoRA via PEFT
- Freeze/unfreeze parameters
- Expose embed_tokens for the projector to target
- Expose generate() for inference
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMWrapper(nn.Module):
    """Model-agnostic wrapper for causal LMs.

    Args:
        model_name_or_path: HuggingFace model identifier or local path.
        torch_dtype: Precision for model loading (default: bfloat16).
        device_map: Device placement strategy (default: "auto").
        lora_config: If provided, dict with PEFT LoraConfig kwargs.
            Example: {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"]}
        freeze: Whether to freeze the base LLM parameters (default: True).
        attn_implementation: Attention implementation (default: "sdpa").
    """

    def __init__(
        self,
        model_name_or_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        lora_config: dict[str, Any] | None = None,
        freeze: bool = True,
        attn_implementation: str = "sdpa",
    ):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if freeze:
            self.freeze_base()

        if lora_config is not None:
            self._apply_lora(lora_config)

    def _apply_lora(self, lora_kwargs: dict[str, Any]) -> None:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(task_type="CAUSAL_LM", **lora_kwargs)
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()

    def freeze_base(self) -> None:
        """Freeze all base model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_base(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the token embedding layer (needed for projector target dim)."""
        return self.model.get_input_embeddings()

    def get_embedding_dim(self) -> int:
        return self.get_input_embeddings().embedding_dim

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> Any:
        """Forward pass using pre-computed input embeddings (from projector).

        Args:
            inputs_embeds: (B, T, D_llm) — combined speech + text embeddings.
            attention_mask: (B, T) attention mask.
            labels: (B, T) target token IDs for loss computation. Use -100 for
                    positions that should be ignored (e.g. speech prefix).

        Returns:
            CausalLMOutputWithPast from the underlying model.
        """
        # Gemma 3 requires token_type_ids during training
        if "token_type_ids" not in kwargs and inputs_embeds is not None:
            kwargs["token_type_ids"] = torch.zeros(
                inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
            )

        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Generate token IDs from speech embeddings.

        Args:
            inputs_embeds: (B, T, D_llm) — speech prefix embeddings.
            attention_mask: (B, T) mask.
            **generate_kwargs: Forwarded to model.generate() (max_new_tokens, etc.)

        Returns:
            Generated token IDs.
        """
        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )