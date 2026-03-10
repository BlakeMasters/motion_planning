from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model


class DecisionTransformer(nn.Module):
    """GPT-style Decision Transformer for continuous action prediction."""

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_size: int,
        max_length: int,
        max_ep_len: int,
        n_layer: int,
        n_head: int,
        dropout: float,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.max_length = max_length

        config = GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=4 * hidden_size,
            activation_function="relu",
            n_positions=3 * max_length + 2,
            resid_pdrop=dropout,
            attn_pdrop=dropout,
        )
        self.transformer = GPT2Model(config)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.predict_action = nn.Sequential(nn.Linear(hidden_size, act_dim), nn.Tanh())

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq = states.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones((bsz, seq), dtype=torch.long, device=states.device)

        t_emb = self.embed_timestep(timesteps)
        s_emb = self.embed_state(states) + t_emb
        a_emb = self.embed_action(actions) + t_emb
        r_emb = self.embed_return(returns_to_go) + t_emb

        stacked = torch.stack([r_emb, s_emb, a_emb], dim=2).reshape(bsz, 3 * seq, self.hidden_size)
        stacked = self.embed_ln(stacked)
        attn = torch.stack([attention_mask] * 3, dim=2).reshape(bsz, 3 * seq)

        out = self.transformer(inputs_embeds=stacked, attention_mask=attn).last_hidden_state
        out = out.reshape(bsz, seq, 3, self.hidden_size)
        return self.predict_action(out[:, :, 1, :])
