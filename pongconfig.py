from dataclasses import dataclass
import pygame
import random

@dataclass
class PongConfig:
    WHITE: tuple =  (255, 255, 255)
    BLACK: tuple  = (0, 0, 0)

    # Screen dimensions
    SCREEN_WIDTH:int = 640
    SCREEN_HEIGHT:int  = 480

    # Paddle dimensions and speed
    PADDLE_WIDTH:int  = 15
    PADDLE_HEIGHT:int = 65
    PADDLE_SPEED:int = 6

    # Ball dimensions and speed
    BALL_WIDTH:int  = 15
    BALL_HEIGHT:int  = 15
    BALL_SPEED_X:int = 5
    BALL_SPEED_Y:int = 7
    MAX_BALL_SPEED: int = 12

    # Misc
    MAX_STEPS: int = 10000

    # GA
    CROSSOVER_RATE: int = 0.60
    MUTATION_RATE: int = 0.1

config = PongConfig()
