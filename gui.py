import pygame
from pongconfig import config

class PongUI:
    def __init__(self, env):
        pygame.init()
        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        pygame.display.set_caption("Pong")
        self.env = env
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 55)

    def render(self):
        # Clear screen
        self.screen.fill(config.BLACK)

        # Draw paddles and ball
        self.env.paddle_a.draw(self.screen)
        self.env.paddle_b.draw(self.screen)
        self.env.ball.draw(self.screen)

        # Display score
        self.display_score()

        pygame.display.flip()
        self.clock.tick(60)

    def display_score(self):
        score_display = self.font.render(self.env.game.get_score(), True, config.WHITE)
        self.screen.blit(score_display, (config.SCREEN_WIDTH // 2 - 50, 10))