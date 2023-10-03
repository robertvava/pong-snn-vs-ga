from pongconfig import config
import random 
import pygame


class Paddle:
    def __init__(self, x, y, height=config.PADDLE_HEIGHT):
        self.x = x
        self.y = y
        self.height = height

    def move(self, direction):
        if direction == "up" and self.y > 0:
            self.y -= config.PADDLE_SPEED
        elif direction == "down" and self.y < config.SCREEN_HEIGHT - self.height:
            self.y += config.PADDLE_SPEED
        

    def draw(self, screen):
        pygame.draw.rect(screen, config.WHITE, (self.x, self.y, config.PADDLE_WIDTH, config.PADDLE_HEIGHT))


class Wall(Paddle):
    def __init__(self, x=config.SCREEN_WIDTH - config.PADDLE_WIDTH, y=0, height=config.SCREEN_HEIGHT):
        self.x = x
        self.y = y
        self.height = height if height else config.PADDLE_HEIGHT

    def move(self, direction):
        return 0 # Walls don't move


    def draw(self, screen):
        pygame.draw.rect(screen, config.WHITE, (self.x, self.y, config.PADDLE_WIDTH, self.height))


class Ball:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed_x = config.BALL_SPEED_X * random.choice([-1, 1])
        self.speed_y = config.BALL_SPEED_Y * random.choice([-1, 1])
        self.counter = 0 
        # self.hit_top = False
        # self.hit_bottom = False
        self.hit_paddle_a = False

    def move(self, paddle_a, paddle_b):
        self.x += self.speed_x
        self.y += self.speed_y
        self.counter += 1

        if self.y <= 0:
            self.speed_y = abs(self.speed_y)  # Ensure vertical speed is positive
            self.y = 0  # Prevent ball from going out of bounds
            self.hit_top = True
        else:
            self.hit_top = False

        if self.y >= config.SCREEN_HEIGHT - config.BALL_HEIGHT:
            self.speed_y = -abs(self.speed_y)  # Ensure vertical speed is negative
            self.y = config.SCREEN_HEIGHT - config.BALL_HEIGHT  # Prevent ball from going out of bounds
            self.hit_bottom = True
        else:
            self.hit_bottom = False

        self.speed_y += random.choice([-1, 1]) * random.uniform(0.05, 0.2)

    def reset(self):
        self.x = config.SCREEN_WIDTH // 2 - config.BALL_WIDTH // 2
        self.y = config.SCREEN_HEIGHT // 2 - config.BALL_HEIGHT // 2
        self.speed_x = config.BALL_SPEED_X * random.choice([-1, 1])
        self.speed_y = config.BALL_SPEED_Y * random.choice([-1, 1])
        self.counter = 0
        
    def draw(self, screen):
        pygame.draw.ellipse(screen, config.WHITE, (self.x, self.y, config.BALL_WIDTH, config.BALL_HEIGHT))
        pygame.draw.aaline(screen, config.WHITE, (config.SCREEN_WIDTH // 2, 0), (config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT))
