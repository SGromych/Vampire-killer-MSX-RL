"""
Накопление разбивки наград за эпизод и экспорт в info / TensorBoard.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class EpisodeRewardLogger:
    """Накопление компонентов награды за эпизод. v3: last_extra для диагностики доминирования."""

    episode_components: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    episode_total: float = 0.0
    step_count: int = 0
    last_extra: dict | None = field(default=None, repr=False)

    def reset(self) -> None:
        self.episode_components = defaultdict(float)
        self.episode_total = 0.0
        self.step_count = 0
        self.last_extra = None

    def add_step(self, reward: float, components: dict[str, float], extra: dict | None = None) -> None:
        self.episode_total += reward
        self.step_count += 1
        for k, v in components.items():
            self.episode_components[k] += v
        if extra is not None:
            self.last_extra = extra

    def get_episode_components(self) -> dict[str, float]:
        return dict(self.episode_components)

    def get_summary(self) -> dict[str, float]:
        """Итог по эпизоду для лога и TensorBoard."""
        out = dict(self.episode_components)
        out["total"] = self.episode_total
        out["steps"] = float(self.step_count)
        return out

    def format_episode_report(self, last_extra: dict | None = None) -> str:
        """Текстовый отчёт в конце эпизода. v3: key/door, при last_extra — предупреждение death_dominance."""
        c = self.get_episode_components()
        total = self.episode_total
        steps = self.step_count
        lines = [
            f"[reward] episode total={total:.3f} steps={steps}",
            f"  step={c.get('step', 0):.3f} pickup={c.get('pickup', 0):.3f} death={c.get('death', 0):.3f} "
            f"novelty={c.get('novelty', 0):.3f} pingpong={c.get('pingpong', 0):.3f} stuck={c.get('stuck', 0):.3f} "
            f"key={c.get('key', 0):.3f} door={c.get('door', 0):.3f}",
        ]
        if last_extra is None:
            last_extra = getattr(self, "last_extra", None)
        if last_extra and last_extra.get("death_dominance_warning"):
            lines.append("  [reward] WARNING: death_penalty > 50% of total reward")
        return "\n".join(lines)
