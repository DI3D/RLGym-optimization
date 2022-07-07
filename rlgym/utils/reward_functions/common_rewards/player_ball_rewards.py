import numpy as np
import math as math_py

from rlgym.utils import RewardFunction, math
from rlgym.utils.common_values import BALL_RADIUS, CAR_MAX_SPEED
from rlgym.utils.gamestates import GameState, PlayerData


class LiuDistancePlayerToBallReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Compensate for inside of ball being unreachable (keep max reward at 1)
        # dist = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
        dist = math.norm_1d([i - j for i, j in zip(player.car_data.position, state.ball.position)]) - BALL_RADIUS
        return math_py.exp(-0.5 * dist / CAR_MAX_SPEED)  # Inspired by https://arxiv.org/abs/2105.12196


class VelocityPlayerToBallReward(RewardFunction):
    def __init__(self, use_scalar_projection=False):
        super().__init__()
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel = player.car_data.linear_velocity
        # pos_diff = state.ball.position - player.car_data.position
        pos_diff = [i - j for i, j in zip(state.ball.position, player.car_data.position)]
        if self.use_scalar_projection:
            # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
            # Max value should be max_speed / ball_radius = 2300 / 92.75 = 24.8
            # Used to guide the agent towards the ball
            inv_t = math.scalar_projection_1d(vel, pos_diff)
            return inv_t
        else:
            # Regular component velocity
            # norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            partial = math.norm_1d(pos_diff)
            # partial = np.linalg.norm(pos_diff)
            norm_pos_diff = [i / partial for i in pos_diff]

            # norm_vel = vel / CAR_MAX_SPEED
            norm_vel = [i / CAR_MAX_SPEED for i in vel]
            # return float(np.dot(norm_pos_diff, norm_vel))
            return sum([i*j for i, j in zip(norm_pos_diff, norm_vel)])


class FaceBallReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # pos_diff = state.ball.position - player.car_data.position
        pos_diff = [i - j for i, j in zip(state.ball.position, player.car_data.position)]
        # norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        partial = math.norm_1d(pos_diff)
        # partial = np.linalg.norm(pos_diff)
        norm_pos_diff = [i / partial for i in pos_diff]
        # return float(np.dot(player.car_data.forward(), norm_pos_diff))
        return sum([i * j for i, j in zip(player.car_data.forward(), norm_pos_diff)])


class TouchBallReward(RewardFunction):
    def __init__(self, aerial_weight=0.):
        self.aerial_weight = aerial_weight

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched:
            # Default just rewards 1, set aerial weight to reward more depending on ball height
            return ((state.ball.position[2] + BALL_RADIUS) / (2 * BALL_RADIUS)) ** self.aerial_weight
        return 0
