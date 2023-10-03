from pongconfig import config
import torch 
from env.objects import Paddle, Ball



class PongEnvironment:
    def __init__(self, paddle_a = Paddle(0, config.SCREEN_HEIGHT // 2 - config.PADDLE_HEIGHT // 2), paddle_b = Paddle(config.SCREEN_WIDTH - config.PADDLE_WIDTH, config.SCREEN_HEIGHT // 2 - config.PADDLE_HEIGHT // 2), game_duration = 10000):
        self.paddle_a = paddle_a
        self.paddle_b = paddle_b
        self.ball = Ball(config.SCREEN_WIDTH // 2 - config.BALL_WIDTH // 2, config.SCREEN_HEIGHT // 2 - config.BALL_HEIGHT // 2)
        self.game = Game()
        self.done = False
        self.game_duration = game_duration
        self.current_step = 0
        self.with_reward = False

    def reset(self):
        self.__init__()
        return self.get_state()

    def encode_state(self, state, prev_state=None):
        ball_y_diff = state["ball_y"] - prev_state["ball_y"] if prev_state else 0
        ball_x_diff = state["ball_x"] - prev_state["ball_x"] if prev_state else 0
        paddle_y_diff = state["paddle_a_y"] - config.SCREEN_HEIGHT // 2

        return torch.tensor([
            state["ball_y"] / config.SCREEN_HEIGHT, 
            state["ball_speed_y"] / config.MAX_BALL_SPEED, 
            state["paddle_a_y"] / config.SCREEN_HEIGHT,
            ball_y_diff / config.SCREEN_HEIGHT,
            ball_x_diff / config.SCREEN_WIDTH,
            paddle_y_diff / config.SCREEN_HEIGHT
        ]).unsqueeze(0)
    
    def get_state(self):
        return {
            "ball_y": self.ball.y,
            "ball_speed_y": self.ball.speed_y,
            "paddle_a_y": self.paddle_a.y,
            "ball_x": self.ball.x
        }

    def step(self, action_a, action_b):


        self.current_step += 1
        
        if action_a == 0:
            self.paddle_a.move("up")
        elif action_a == 1:
            self.paddle_a.move("down")
        elif action_a == 2:
            pass

        if action_b == 0:
            self.paddle_b.move("up")
        elif action_b == 1:
            self.paddle_b.move("down")
        
        
        self.ball.move(self.paddle_a, self.paddle_b)


        # Ball collision with paddles
        if self.ball.y + config.BALL_HEIGHT > self.paddle_a.y and self.ball.y < self.paddle_a.y + config.PADDLE_HEIGHT and self.ball.x <= config.PADDLE_WIDTH:
            self.ball.speed_x = -self.ball.speed_x
            self.ball.hit_paddle_a = True
        else:
            self.ball.hit_paddle_a = False
        
       
        # Collision detection for paddle B (either wall or regular paddle)
        if self.ball.x >= config.SCREEN_WIDTH - config.PADDLE_WIDTH - config.BALL_WIDTH:
            if self.ball.y + config.BALL_HEIGHT > self.paddle_b.y and self.ball.y < self.paddle_b.y + self.paddle_b.height:
                self.ball.speed_x = -self.ball.speed_x

        reward = 0

        if self.ball.hit_paddle_a:
            reward_hit_paddle = 1
        else:
            reward_hit_paddle = 0

        if self.ball.x <= 0:
            self.game.update_score("b")
            self.ball.reset()  
            score_penalty = 2
        else: 
            score_penalty = 0

        if self.ball.x >= config.SCREEN_WIDTH:
            self.game.update_score("a")
            self.ball.reset()  
            reward = 1         # Redundant, but good to keep for miscellanous ideas!!! 
        
        if self.current_step == self.game_duration:
            self.done = True
        

        reward = reward_hit_paddle - score_penalty

        if self.with_reward:
            return self.encode_state(self.get_state()), reward, self.done
        
        return self.get_state(), self.done


class Game:
    def __init__(self):
        self.score_a = 0
        self.score_b = 0

    def update_score(self, scorer):
        if scorer == "a":
            self.score_a += 1
        else:
            self.score_b += 1

    def get_score(self):
        return f"{self.score_a} - {self.score_b}"