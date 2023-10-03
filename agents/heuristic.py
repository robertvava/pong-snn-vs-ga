from pongconfig import config


class HeuristicAgent:
    def __init__(self, paddle):
        self.paddle = paddle

    def get_action(self, state, encoded_state):

        ball_y = state["ball_y"]
        if ball_y > self.paddle.y:
            return 1  # Move down
        else:
            return 0  # Move up
