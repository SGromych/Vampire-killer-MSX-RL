from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pygame

from .env import (
    ACTION_NOOP,
    ACTION_RIGHT,
    ACTION_LEFT,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_ATTACK,
    ACTION_RIGHT_JUMP,
    ACTION_LEFT_JUMP,
    ACTION_RIGHT_JUMP_ATTACK,
    ACTION_LEFT_JUMP_ATTACK,
)


@dataclass
class HumanControllerConfig:
    fps: int = 10


class HumanController:
    """
    Слой, собирающий события pygame и возвращающий дискретное действие.

    Esc -> None (сигнал завершения эпизода/запуска)
    """

    def __init__(self, cfg: HumanControllerConfig):
        self.cfg = cfg
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((320, 240))
        pygame.display.set_caption("Vampire Killer — Human Demo")

    def close(self) -> None:
        pygame.quit()

    def poll_action(self) -> Optional[int]:
        """
        Блокируется на один шаг (1 / fps), обрабатывает события и возвращает action id.
        """
        action = ACTION_NOOP
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None

        keys = pygame.key.get_pressed()

        left = keys[pygame.K_LEFT]
        right = keys[pygame.K_RIGHT]
        up = keys[pygame.K_UP]
        down = keys[pygame.K_DOWN]
        attack_key = keys[pygame.K_z] or keys[pygame.K_SPACE]
        jump_key = keys[pygame.K_x]  # если нужен отдельный прыжок

        # Комбинации приоритетнее одиночных направлений
        if (right and up) or (right and jump_key):
            if attack_key:
                action = ACTION_RIGHT_JUMP_ATTACK
            else:
                action = ACTION_RIGHT_JUMP
        elif (left and up) or (left and jump_key):
            if attack_key:
                action = ACTION_LEFT_JUMP_ATTACK
            else:
                action = ACTION_LEFT_JUMP
        elif right:
            action = ACTION_RIGHT
        elif left:
            action = ACTION_LEFT
        elif down:
            action = ACTION_DOWN
        elif up or jump_key:
            action = ACTION_UP
        elif attack_key:
            action = ACTION_ATTACK
        else:
            action = ACTION_NOOP

        self.clock.tick(self.cfg.fps)
        return action

