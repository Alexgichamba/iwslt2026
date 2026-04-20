#--------------------------------LLama 1 8B training script-----------------------------
import os
import math 
import time
import inspect 
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
from torch.nn import functional as F
#import tiktoken
# from data.hellaswag import render_example, iterate_examples
from st.models.kvcache import KVcache 
from transformers import AutoModelForCausalLM
import numpy as np
# from torch._dynamo import optimize
import random
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

# from torch.nn.attention import SDPBackend, sdpa_kernel
#---------------------------------------------------------------------------------------

block_size = 2048                       # embedding dimension (hidden size) , 4096 for LLaMA 1 8B
intermediate_size =  5120               # feedforward intermediate size
vocab_size = 49152                      # vocabulary size of LLaMA models      change to 50304 , 32000 for llama 3
n_layers = 36                           # number of transformer layers , 32 for LLaMA 1 8B
n_heads = 32                            # number of attention heads, 32 for LLaMA 1 8B
n_kv_heads = 8                          # number of key/value heads for multi-head attention, usually n_kv_heads = 8
rope_theta = 500000                     # RoPE base frequency
max_seq_len =  2048                     # maximum supported sequence length,, 2048 for LLaMA 1 8B
multiple_of = 256                       # make SwiGLU hidden size multiple of large power of 2
norm_eps = 1e-5                        # layernorm epsilon change to 1e-5
# max_batch_size = 16                    # maximum batch size for training change to 32 for llama 3 config


#***********************************************************************************************
@dataclass
class ModelArgs:
    dim: int = block_size
    intermediate_size: int = intermediate_size
    vocab_size: int = vocab_size
    n_layers: int = n_layers
    n_heads: int = n_heads
    n_kv_heads: int = n_kv_heads
    rope_theta: int = rope_theta
    multiple_of: int = multiple_of
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = norm_eps

    # max_batch_size: int = max_batch_size
    max_seq_len: int = max_seq_len

#***********************************************************************************************
class LLamaRMSNorm(nn.Module):
    ''' 
    https://arxiv.org/abs/1910.07467
    Root Mean Square Layer Normalization to enable stable training of LLaMA models.
    From the LLaMA paper: The successs of the llayer normalization was hypothesized that we do not 
    need to center the activations, and that normalizing the variance is sufficient for training stability. Thus,
    we hypothesize that the re-scaling invariance is the reason for success of LayerNorm, rather than re-centering invariance.
    '''

    def __init__ (self, hidden_size, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    def forward (self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states.to(input_dtype)
        return self.weight * hidden_states
    # similar to the implemenation in T5layerNorm


#***********************************************************************************************
class LlamaRotaryEmbedding(nn.Module):
    """
    https://aclanthology.org/N18-2074/ Relative Positional Encoding for Transformers
    https://arxiv.org/abs/2104.09864  Rotary Position Embedding

    read blog to understand how to fix the frequency of rope 
    https://freedium.cfd/https://medium.com/@hugmanskj/mastering-llama-understanding-rotary-positional-embedding-rope-91d1e707e95a
    llama 3 paper : We increase the RoPE base frequency hyperparameter to 500,000. 
    This enables us to better support longer contexts; Xiong et al. (2023) showed this value to be effective for context lengths up to 32,768.
    sΘ={θi=10000−2(i−1)/d,i∈[1,2,...,d/2]}.
    theta_i = base^ ( -2 * (i-1) / dim ) for i = {1,2,...,dim/2}
    
    """ 
    def __init__(self,dim, max_position_embeddings=2048, base=500000, device=None, scaling_factor=1.0): # change value from 10000 to 500000
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.dim = dim
        self.scaling_factor = scaling_factor
        self.base = base
        # inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # for BC we register the cos and sin cache
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
            # x: [bs, num_attention_heads, seq_len, head_size]
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            position_ids_expanded = position_ids[:, None, :].float()
            # Force float32 since bfloat16 loses precision on long contexts
            # See https://github.com/huggingface/transformers/pull/29285
            device_type = x.device.type
            device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):
                # freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                freqs = torch.einsum("i, b j -> b j i", self.inv_freq, position_ids.float())
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def rotate_half(self, x):
        """
        Rotary Embedding helper function that rotates half the hidden dims
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids = None, unqueeze_dim = 1):
        """
        Apply Rotary Embedding to query and key tensors.
        Args:

        q[torch.Tensor]: query tensor of shape [bs, num_attention_heads, seq_len, head_size]
        k[torch.Tensor]: key tensor of shape [bs, num_attention_heads, seq_len, head_size]
        cos[torch.Tensor]: precomputed cosines of shape [bs, seq_len, head_size]
        sin[torch.Tensor]: precomputed sines of shape [bs, seq_len, head_size]
        position_ids[torch.Tensor]: position ids of shape [bs, seq_len]
            position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
            unsqueeze_dim[int]: dimension to unsqueeze cos and sin tensors

        Returns:
        tuple(torch.Tensor): comprising the query and key tensors after applying rotary embeddings.
   
        """
        cos = cos.unsqueeze(unqueeze_dim)
        sin = sin.unsqueeze(unqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads for multi-head attention when num_kv_heads < num_heads.
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)

    """

    batch, n_kv_heads, seqlen, head_dim = hidden_states.shape

    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seqlen, head_dim)
    return hidden_states.reshape(batch, n_kv_heads * n_rep, seqlen, head_dim)



#***********************************************************************************************
class LlamaAttention(nn.Module):
    """
    Starting from LLama 2 (70B) and later models they used Grouped Query Attention (GQA).
    However in this implementation we only implement standard multi-head self-attention as in LLama 1.
    
    later optimizer this to implement GQA. https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/04_gqa/gpt_with_kv_gqa.py
    GQA paper: https://arxiv.org/abs/2305.13245
    
    """
    def __init__(self, config: ModelArgs, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.dim
        self.num_heads = config.n_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.n_kv_heads
        self.n_kv_groups = self.num_heads // self.num_kv_heads
        self.max_postion_embeddings = config.max_seq_len
        self.rope_theta = config.rope_theta
        self.is_causal = True


        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                    f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                    f" and `num_heads`: {self.num_heads})."
                )


        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        self.rope = LlamaRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.max_postion_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self, 
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        cache: KVcache = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.shape
        
        # Project
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to [bsz, seqlen, num_heads, head_dim] 
        # Note: We keep seqlen at dim 1 for RoPE and Flash Attn compatibility
        query_states = query_states.view(bsz, seqlen, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        # 1. Generate RoPE frequencies
        # Adjust rope input: it usually expects [bsz, num_heads, seqlen, head_dim] or handled inside apply_rotary
        cos, sin = self.rope(value_states, position_ids)

        # 2. Apply RoPE to Query and the CURRENT Key shard
        # Transpose to [bs, heads, seq, dim] for the apply_rotary_pos_emb function
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        
        query_states, key_states = self.rope.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # 3. Handle KV Cache (Concatenate AFTER RoPE for the new token)
        if use_cache and cache is not None:
            value_states = value_states.transpose(1, 2) # [bs, kv_heads, seq, dim]
            
            if len(cache.key_list[self.layer_idx]) > 0:
                key_states_old, value_states_old = cache.get_key_values(self.layer_idx)
                key_states = torch.cat([key_states_old, key_states], dim=2)
                value_states = torch.cat([value_states_old, value_states], dim=2)
            
            cache.update(self.layer_idx, key_states, value_states)
        else:
            value_states = value_states.transpose(1, 2)

        # 4. Grouped Query Attention (Repeat KV heads)
        key_states = repeat_kv(key_states, self.n_kv_groups)
        value_states = repeat_kv(value_states, self.n_kv_groups)

        # 5. Attention Computation
        # If prefilling (seqlen > 1), we MUST use causal masking so prompt tokens don't attend to future prompt tokens.
        # If decoding (seqlen == 1), causal masking is irrelevant but SDPA/Flash might expect specific inputs.
        is_causal = seqlen > 1

        if FLASH_ATTN_AVAILABLE and query_states.is_cuda:
            # Flash Attention 2 requires [batch, seq, heads, dim]
            q = query_states.transpose(1, 2)
            k = key_states.transpose(1, 2)
            v = value_states.transpose(1, 2)
            
            attn_output = flash_attn_func(q, k, v, dropout_p=0.0, causal=is_causal)
            attn_output = attn_output.reshape(bsz, seqlen, self.hidden_size)
        else:
            # standard SDPA expects [bs, heads, seq, dim]
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, is_causal=is_causal
            )
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden_size)

        return self.o_proj(attn_output)
    # # Adapted for LLamaAttention.forward 
    # def forward(
    #     self, 
    #     hidden_states: torch.Tensor,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     use_cache: bool = False,
    #     cache: KVcache = None,
    # ) -> torch.Tensor:
    #     bsz, seqlen, _ = hidden_states.shape # [batch_size, seq_len, hidden_size]
    #     query_states = self.q_proj(hidden_states)  # [bs, seq_len, num_heads * head_dim]
    #     key_states = self.k_proj(hidden_states)    # [bs, seq_len, num_kv_heads * head_dim]
    #     value_states = self.v_proj(hidden_states)  # [bs, seq_len, num_kv_heads * head_dim]
        
    #     query_states = query_states.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)  # [bs, num_heads, seq_len, head_dim]
    #     key_states = key_states.view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)      # [bs, num_kv_heads, seq_len, head_dim]
    #     value_states = value_states.view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [bs, num_kv_heads, seq_len, head_dim]


    #     cos, sin = self.rope(value_states, position_ids)  # [bs, seq_len, head_dim]

    #     # Apply RoPE to query and key states
    #     query_states, key_states = self.rope.apply_rotary_pos_emb(
    #         query_states, key_states, cos, sin)

    #     key_states = repeat_kv(key_states, self.n_kv_groups)
    #     value_states = repeat_kv(value_states, self.n_kv_groups)


    #     is_causal = True
    #     if use_cache:
    #         if len(cache.key_list[self.layer_idx]) > 0:

    #             key_states_old , value_states_old = cache.get_key_value(self.layer_idx)
    #             key_states = torch.cat([key_states_old, key_states], dim=2)
    #             value_states = torch.cat([value_states_old, value_states], dim=2)
    #             is_causal = False
    #         cache.update(self.layer_idx, key_states, value_states)

        
    #     # Flash attention if available, fall back to standard attention
    #     if FLASH_ATTN_AVAILABLE and query_states.is_cuda and not use_cache:
    #         # flash_attn requires layout [batch_size, seq_len, num_heads, head_dim] in float16/bfloat16
    #         q = query_states.transpose(1, 2) # [bsz, seqlen, num_heads, head_dim]
    #         k = key_states.transpose(1, 2)   # [bsz, seqlen, num_kv_heads, head_dim]
    #         v = value_states.transpose(1, 2) # [bsz, seqlen, num_kv_heads, head_dim]
            
    #         # The causal flag is automatically handled correctly based on seqlen
    #         attn_output = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=is_causal)
    #         # attn_output is [bsz, seqlen, num_heads, head_dim]
    #         attn_output = attn_output.transpose(1, 2) # [bsz, num_heads, seqlen, head_dim]
    #     elif hasattr(torch.nn.functional, 'scaled_dot_product_attention') and query_states.is_cuda:
    #         # Replaces the deprecated torch.backends.cuda.sdp_kernel
    #         from torch.nn.attention import sdpa_kernel, SDPBackend
    #         with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
    #             attn_output = torch.nn.functional.scaled_dot_product_attention(
    #                 query_states,
    #                 key_states,
    #                 value_states,
    #                 is_causal=is_causal,
    #             )  # [bs, num_heads, seq_len, head_dim]
    #     else:
    #         attn_output = torch.nn.functional.scaled_dot_product_attention(
    #             query_states,
    #             key_states,
    #             value_states,
    #             is_causal=is_causal,
    #         )  # [bs, num_heads, seq_len, head_dim]

    #     attn_output = attn_output.transpose(1, 2).contiguous()  # [bs, seq_len, num_heads * head_dim]
    #     attn_output = attn_output.view(bsz, seqlen, -1)  # [bs, seq_len, hidden_size]
    #     attn_output = self.o_proj(attn_output)  # [bs, seq_len, hidden_size]
    #     return attn_output



#***********************************************************************************************
class LLamaMLP(nn.Module):
    """
    LLaMA Feedforward network with SwiGLU activation
    
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hidden_size = config.dim
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.activation = nn.SiLU()  # SwiGLU activation


    def forward(self,x):
        # Use bfloat16 for intermediate computations
        # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Fuse activation and multiplication correctly (SwiGLU: SiLU(gate) * up)
        hidden = self.activation(gate) * up
        # del gate, up  # Free memory
        return self.down_proj(hidden)



#***********************************************************************************************
class Block(nn.Module):
    def __init__(self, layer_id : int, config: ModelArgs):
        super().__init__()
        self.hidden_state = config.dim
        self.attn = LlamaAttention(config, layer_idx=layer_id)
        self.mlp = LLamaMLP(config)
        self.input_layernorm = LLamaRMSNorm(config.dim, eps=config.norm_eps)
        self.post_attention_layernorm = LLamaRMSNorm(config.dim, eps=config.norm_eps)



    def forward(
            self, 
            hidden_states: torch.Tensor,
            position_ids: Optional[torch.LongTensor] = None,
            use_cache: bool = False,
            cache: KVcache = None,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self-Attention
        hidden_states = self.attn(
            hidden_states =hidden_states,
            position_ids=position_ids,
            use_cache=use_cache,
            cache=cache
        )

        hidden_states = residual + hidden_states


        # Full Connected layer

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = hidden_states
        return outputs



#***********************************************************************************************
class LlamaModel(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.config = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.embed_tokens = nn.Embedding(

            params.vocab_size, params.dim   
        )

        # create layers
        self.layers = nn.ModuleList()

        for layer_id in range(params.n_layers):
            self.layers.append(Block(layer_id, params))


        self.norm = LLamaRMSNorm(params.dim, eps=params.norm_eps)

    def forward( self , tokens: torch.Tensor | None = None, position_ids : torch.LongTensor | None = None, use_cache: bool = False, cache: KVcache = None, inputs_embeds: torch.Tensor | None = None):
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.embed_tokens(tokens)
        
        def create_custom_forward(layer):
            def custom_forward(*args):
                return layer(*args)
            return custom_forward

        for i, layer in enumerate(self.layers):
            if not use_cache:
                h = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    h,
                    position_ids,
                    use_cache,
                    cache,
                    use_reentrant=False
                )
            else:
                h = layer(
                    h,
                    position_ids,
                    use_cache=use_cache,
                    cache=cache
                )
        h = self.norm(h)
        return h

#***********************************************************************************************
class LlamaTransformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        # Add gradient checkpointing attribute
        self.gradient_checkpointing = False

        # init parameters
        self.apply(self._init_weights)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for the model."""
        self.gradient_checkpointing = True
        self.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the model."""
        self.gradient_checkpointing = False
        self.model.gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # @torch._dynamo.disable
    def forward(
        self, 
        input_ids: torch.LongTensor | None = None,
        targets: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        cache: Optional[KVcache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ): 
        # Use CUDA graphs for repeated forward passes if possible
        if inputs_embeds is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
            device = inputs_embeds.device
        else:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
        static_input = (batch_size == 8 and seq_len == 2048 and not use_cache)  # Adjust based on your B and T
        
        if use_cache:
            if len(cache.key_list[0]) > 0:
                start_pos = cache.key_list[0].shape[2]
                position_ids = torch.arange(
                    start_pos,
                    start_pos + seq_len,
                    dtype=torch.long,
                    device=input_ids.device,
                ).unsqueeze(0).expand(batch_size, seq_len)


            else: 
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        else:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        
        outputs = self.model(
            tokens=input_ids,
            position_ids = position_ids,
            use_cache=use_cache,
            cache=cache,
            inputs_embeds=inputs_embeds,
        )  # [bs, seq_len, hidden_size]
        #43476 , 43477

        hidden_states = outputs
        logits = self.lm_head(hidden_states)  # [bs, seq_len, vocab_size]
        logits = logits.float()  # convert to float32 for numerical stability
        if targets is None:
            return logits
        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

        return logits, loss

    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.LongTensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ):
        """
        Fast token generation using KV caching.
        """
        from kvcache import KVcache
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize Cache
        cache = KVcache(self.config.n_layers)
        
        # 1. Prefill step
        logits = self.forward(input_ids, use_cache=True, cache=cache)
        
        # 2. Extract out the last token logits
        next_token_logits = logits[:, -1, :]
        
        # Temperature & Greedy decoding for baseline
        if temperature != 1.0 and temperature > 0:
            next_token_logits = next_token_logits / temperature
        
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
        generated_tokens = [next_token]
        
        # Create an early stopping mask
        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # 3. Decoding Loop
        for _ in range(max_new_tokens - 1):
            logits = self.forward(next_token, use_cache=True, cache=cache)
            next_token_logits = logits[:, -1, :]
            
            if temperature != 1.0 and temperature > 0:
                next_token_logits = next_token_logits / temperature
                
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            
            if eos_token_id is not None:
                # If a sequence hits eos_token_id, we replace subsequent tokens with eos_token_id (or pad)
                # and if all sequences are finished, we can break early
                is_eos = (next_token.squeeze(1) == eos_token_id)
                next_token[~unfinished_sequences] = eos_token_id
                unfinished_sequences = unfinished_sequences & ~is_eos
                
            generated_tokens.append(next_token)
            
            if eos_token_id is not None and not unfinished_sequences.any():
                break
                
        # Concatenate generated tokens and attach to input context
        generated_tensor = torch.cat(generated_tokens, dim=1)
        return torch.cat([input_ids, generated_tensor], dim=1)



    @classmethod
    def from_custom_pretrained(cls,ckp_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ckp = torch.load(ckp_path, map_location=device)
        config = ckp['config']
        model = LlamaTransformer(config)
        model.load_state_dict(ckp['model'])
        return model

    @classmethod
    def from_pretrained(cls,model_type):
        print("loading weights from pretrained model... ", model_type )
        model_hf = AutoModelForCausalLM.from_pretrained(model_type)
        
        dim = model_hf.config.hidden_size
        intermediate_size = model_hf.config.intermmediate_size
        n_layers = model_hf.config.num_hidden_layers
        n_heads = model_hf.config.num_attention_heads
        n_kv_heads = model_hf.config.num_key_value_heads 
        vocab_size = model_hf.config.vocab_size
        rope_theta = model_hf.config.rope_theta
        max_seq_len = model_hf.config.max_position_embeddings



        config = ModelArgs(
            dim=dim,
            intermediate_size=intermediate_size,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            vocab_size=vocab_size,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
        )
        model = LlamaTransformer(config)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if k.endswith('bias')] # discard bias terms
        print("total parameters to load: ", len(sd_keys))

        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type,master_process):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
# def load_tokens(filename, max_tokens=None):
#     if filename.endswith('.npy'):
#         data = np.load(filename)
#     else:
#         data = np.fromfile(filename, dtype=np.uint16)
#     if max_tokens is not None:
#         data = data[:max_tokens]
#     data = data.astype(np.int32)
#     return torch.from_numpy(data).long()

'''
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root, master_process,
                 max_shards_per_lang=None, val_tokens_per_lang=10_000_000):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.data_root = data_root
        self.master_process = master_process
        self.val_tokens_per_lang = val_tokens_per_lang
        assert split in {'train', 'val'}

        has_bin = any(f.endswith('.bin') for _, _, files in os.walk(self.data_root) for f in files)

        if has_bin:
            from collections import defaultdict
            lang_shards = defaultdict(list)
            for root, dirs, files in os.walk(self.data_root):
                bin_files = sorted(f for f in files if f.endswith('.bin'))
                if bin_files:
                    lang = os.path.basename(root)
                    for f in bin_files:
                        lang_shards[lang].append(os.path.join(root, f))

            # If we see our manually mixed structure, use it exclusively
            if 'mixed_train' in lang_shards and 'val_holdout' in lang_shards:
                if split == 'train':
                    shards = lang_shards['mixed_train']
                    if max_shards_per_lang is not None:
                        shards = shards[:max_shards_per_lang]
                else:
                    shards = lang_shards['val_holdout']
            else:
                shards = []
                for lang, shard_list in sorted(lang_shards.items()):
                    # if len(shard_list) > 1: 
                    if split == 'val':
                        shards.append(shard_list[-1])
                    else: 
                        # shards.append(shard_list[0])
                        train_shards = shard_list[:-1] if len(shard_list) > 1 else shard_list
                        shards.extend(train_shards)
                # else:
                    #     train_list = shard_list[:-1]
                    #     if max_shards_per_lang is not None:
                    #         train_list = train_list[:max_shards_per_lang]
                    #     shards.extend(train_list)
                random.seed(42)
                random.shuffle(shards)
                self.shards = shards[process_rank::num_processes]
            if self.master_process:
                n_langs = len(lang_shards)
                if split == 'val':
                    total_val_tok = n_langs * val_tokens_per_lang
                    print(f"[DataLoader] val: {len(shards)} shards, "
                          f"using first {val_tokens_per_lang/1e6:.0f}M tokens/lang "
                          f"= {total_val_tok/1e6:.0f}M tokens total")
                else:
                    total_train_tok = len(shards) * 100_000_000
                    note = f" [sample: max {max_shards_per_lang}/lang]" if max_shards_per_lang else ""
                    print(f"[DataLoader] train: {len(shards)} shards across {n_langs} languages "
                          f"(~{total_train_tok/1e9:.2f}B tokens){note}")
        else:
            shards = []
            for root, dirs, files in os.walk(self.data_root):
                for f in files:
                    if f.endswith('.npy') and split in f:
                        shards.append(os.path.join(root, f))
            if self.master_process:
                print(f"[DataLoader] {split}: {len(shards)} .npy shards in {self.data_root}")

        self.shards = sorted(shards)
        self.split = split
        assert len(self.shards) > 0, f"no shards found for split '{split}' in {data_root}"
        self.reset()

    def _load_current_shard(self):
        if self.split == 'val':
            return load_tokens(self.shards[self.current_shard], max_tokens=self.val_tokens_per_lang)
        return load_tokens(self.shards[self.current_shard])

    def reset(self):
        self.current_shard = 0
        self.tokens = self._load_current_shard()
        self.current_position = self.B * self.T * self.process_rank

    def set_shard_and_pos(self, shard, position):
        self.current_shard = shard
        self.tokens = self._load_current_shard()
        self.current_position = position


    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # store current position before advancing
        current_shard = self.current_shard
        current_position = self.current_position
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            # When we hit a new shard, log it on the master process for visibility
            if self.master_process:
                print(f"[DataLoader] Advancing to shard {self.current_shard}: {os.path.basename(self.shards[self.current_shard])}")
                
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y, current_shard, current_position


'''
import os
import glob
import random
import torch
import numpy as np

def load_tokens(filename):
    npt = np.fromfile(filename, dtype=np.uint16)
    ptt = torch.from_numpy(npt).to(torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root, master_process, max_shards_per_lang=None):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.data_root = data_root
        self.master_process = master_process
        self.max_shards_per_lang = max_shards_per_lang
        assert split in {'train', 'val'}

        # 1. Directly target the correct folders based on your mixing script
        if split == 'val':
            # This captures: val_bem_Latn.bin, val_eng_Latn.bin, etc.
            shard_pattern = os.path.join(data_root, "val_holdout", "val_*.bin")
        else:
            # This captures: shard_00000.bin, shard_00001.bin, etc.
            shard_pattern = os.path.join(data_root, "mixed_train", "shard_*.bin")

        all_shards = sorted(glob.glob(shard_pattern))
        
        if not all_shards:
            raise RuntimeError(f"Could not find any .bin shards in {shard_pattern}")

        # 2. Shuffle for training only (Validation should stay sorted for consistent logging)
        if split == 'train':
            random.seed(42)
            random.shuffle(all_shards)

        # 3. DDP Slicing: Each GPU gets its own subset of the list for training
        if split == 'train':
            self.shards = all_shards[process_rank::num_processes]
        else:
            # CRITICAL DDP FIX: Every rank MUST evaluate all languages to prevent collective mismatch.
            # If Rank 0 evaluates more languages than Rank 1, its dist.all_reduce calls will collide
            # with Rank 1's parameter gradients during the next training step backward pass.
            self.shards = all_shards
        
        if self.master_process:
            print(f"[DataLoader] {split} loaded {len(self.shards)} shards for rank {process_rank}")
            if split == 'val':
                # Helpful to see which languages are actually loaded
                langs = [os.path.basename(s).replace('val_', '').replace('.bin', '') for s in self.shards]
                print(f"[DataLoader] Validation languages found: {', '.join(langs)}")

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def set_shard_and_pos(self, shard_idx, pos):
        """Used for resuming training from a specific checkpoint."""
        self.current_shard = shard_idx % len(self.shards)
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = pos

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        
        # Capture current state before advancing
        shard_idx = self.current_shard
        pos = self.current_position

        self.current_position += B * T
        
        # Advance shard if we reached the end of the current one
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
            
        return x, y, shard_idx, pos