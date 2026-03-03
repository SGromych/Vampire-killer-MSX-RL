"""
Actor-Critic для PPO: shared encoder (как BCNet) + policy head + value head.
Опционально: LSTM после encoder для POMDP (--recurrent). Поддерживает инициализацию из BC.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn

from .env import NUM_ACTIONS
from .bc_model import BCNet, BCNetDeep, FRAME_STACK


def _enc_size(arch: str) -> int:
    return 3136 if arch == "default" else 3200


class ActorCritic(nn.Module):
    """
    Actor-Critic для дискретного action space.
    Encoder — как BCNet (или BCNetDeep при arch=deep).
    При recurrent=True между encoder и головами вставлен LSTM; скрытое состояние сбрасывается на done.
    """

    def __init__(
        self,
        num_actions: int = NUM_ACTIONS,
        in_channels: int = FRAME_STACK,
        hidden: int = 512,
        arch: str = "default",  # "default" | "deep"
        recurrent: bool = False,
        lstm_hidden_size: int = 256,
    ):
        super().__init__()
        self._num_actions = num_actions
        self._in_channels = in_channels
        self._arch = arch
        self._recurrent = recurrent
        self._lstm_hidden_size = lstm_hidden_size

        if arch == "deep":
            base = BCNetDeep(num_actions=num_actions, in_channels=in_channels)
        else:
            base = BCNet(num_actions=num_actions, in_channels=in_channels)

        self.encoder = nn.Sequential(base.conv, nn.Flatten())
        enc_size = _enc_size(arch)

        if recurrent:
            self.lstm = nn.LSTM(enc_size, lstm_hidden_size, batch_first=True, num_layers=1)
            head_in = lstm_hidden_size
        else:
            self.lstm = None
            head_in = enc_size

        self.actor = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (logits, value) или при recurrent (logits, value, h_out, c_out).
        hidden = (h, c) с shape (1, B, lstm_hidden_size).
        """
        h = self.encoder(x)  # (B, enc_size)
        if self.lstm is not None:
            # (B, 1, enc_size) для batch_first LSTM
            h_seq = h.unsqueeze(1)
            out, (h_n, c_n) = self.lstm(h_seq, hidden)  # out (B, 1, lstm_hidden), h_n/c_n (1, B, H)
            h = out.squeeze(1)  # (B, lstm_hidden)
            logits = self.actor(h)
            value = self.critic(h)
            return logits, value, h_n, c_n
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value

    def get_action(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
        hidden: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Sample action, return (action, log_prob, value, next_hidden).
        При non-recurrent next_hidden всегда None.
        """
        if self.lstm is not None:
            logits, value, h_n, c_n = self.forward(x, hidden)
            next_hidden = (h_n, c_n)
        else:
            logits, value = self.forward(x)
            next_hidden = None
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1).item()
            log_prob = dist.log_prob(torch.tensor(action, device=logits.device))
        else:
            action_t = dist.sample()
            action = action_t.item()
            log_prob = dist.log_prob(action_t)
        return action, log_prob, value.squeeze(-1), next_hidden

    def zero_hidden(self, batch_size: int = 1, device: torch.device | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Нулевое скрытое состояние для LSTM. (1, B, H). При non-recurrent не используется."""
        if self.lstm is None:
            raise RuntimeError("zero_hidden only for recurrent model")
        d = device or next(self.parameters()).device
        h = torch.zeros(1, batch_size, self._lstm_hidden_size, device=d, dtype=next(self.parameters()).dtype)
        c = torch.zeros(1, batch_size, self._lstm_hidden_size, device=d, dtype=next(self.parameters()).dtype)
        return h, c

    @property
    def recurrent(self) -> bool:
        return self._recurrent

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def arch(self) -> str:
        return self._arch

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def arch(self) -> str:
        return self._arch


def load_ppo_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> ActorCritic:
    """Загрузить ActorCritic из чекпоинта. Поддерживает recurrent/lstm_hidden_size из чекпоинта."""
    path = Path(path)
    try:
        raw = torch.load(path, map_location=device, weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        raw = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(raw, dict) or "state_dict" not in raw:
        raise ValueError(f"Неверный формат чекпоинта: {path}")
    ckpt = raw
    state = ckpt["state_dict"]
    in_ch = ckpt.get("frame_stack", FRAME_STACK)
    arch = ckpt.get("arch", "default")
    recurrent = ckpt.get("recurrent", False)
    lstm_hidden_size = ckpt.get("lstm_hidden_size", 256)
    model = ActorCritic(
        num_actions=NUM_ACTIONS,
        in_channels=in_ch,
        arch=arch,
        recurrent=recurrent,
        lstm_hidden_size=lstm_hidden_size,
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


def init_from_bc(ac: ActorCritic, bc_path: str | Path) -> None:
    """
    Инициализировать encoder и (при non-recurrent) actor из BC чекпоинта.
    При recurrent копируется только encoder (головы питаются от LSTM, размер входа другой).
    """
    bc_path = Path(bc_path)
    try:
        raw = torch.load(bc_path, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(bc_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        bc_state = raw["state_dict"]
    else:
        bc_state = raw

    ac_state = ac.state_dict()
    for k, v in bc_state.items():
        if k.startswith("conv."):
            ac_k = "encoder.0." + k[5:]
            if ac_k in ac_state and ac_state[ac_k].shape == v.shape:
                ac_state[ac_k] = v
        elif not ac.recurrent and k.startswith("fc.1."):
            ac_k = "actor.0." + k.split(".", 2)[-1]
            if ac_k in ac_state and ac_state[ac_k].shape == v.shape:
                ac_state[ac_k] = v
        elif not ac.recurrent and (k.startswith("fc.3.") or k.startswith("fc.4.")):
            ac_k = "actor.1." + k.split(".", 2)[-1]
            if ac_k in ac_state and ac_state[ac_k].shape == v.shape:
                ac_state[ac_k] = v
    ac.load_state_dict(ac_state, strict=False)
