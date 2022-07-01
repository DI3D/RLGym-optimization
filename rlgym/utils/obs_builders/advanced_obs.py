import math
import numpy as np
from typing import Any, List
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject
from rlgym.utils.obs_builders import ObsBuilder


class AdvancedObs(ObsBuilder):
    POS_STD = 2300  # If you read this and wonder why, ping Rangler in the dead of night.
    ANG_STD = math.pi

    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:

        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads
        # TODO make function in math.py that automatically does sub/divide/add/etc. list comprehension for the user

        # obs = [[i / self.POS_STD for i in ball.position],
        #        # ball.position / self.POS_STD
        #        [i / self.POS_STD for i in ball.linear_velocity],
        #        # ball.linear_velocity / self.POS_STD
        #        [i / self.POS_STD for i in ball.angular_velocity],
        #        # ball.angular_velocity / self.POS_STD
        #        previous_action,
        #        pads]

        obs = [i / self.POS_STD for i in ball.position]
        obs.extend([i / self.POS_STD for i in ball.linear_velocity])
        obs.extend([i / self.POS_STD for i in ball.angular_velocity])
        obs.extend(previous_action)
        obs.extend(pads)

        player_car = self._add_player_to_obs(obs, player, ball, inverted)

        allies = []
        enemies = []

        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            other_car = self._add_player_to_obs(team_obs, other, ball, inverted)

            # Extra info
            team_obs.extend(
                # (other_car.position - player_car.position) / self.POS_STD,
                [i / self.POS_STD for i in [i - j for i, j in zip(other_car.position, player_car.position)]],
                # (other_car.linear_velocity - player_car.linear_velocity) / self.POS_STD
            )
            team_obs.extend([i / self.POS_STD for i in [i - j for i, j in zip(other_car.linear_velocity,
                                                                              player_car.linear_velocity)]])

        obs.extend(allies)
        obs.extend(enemies)
        # this is to flatten, change this somehow?
        return np.fromiter(obs, dtype=np.float32, count=len(obs))

    def _add_player_to_obs(self, obs: List, player: PlayerData, ball: PhysicsObject, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        # rel_pos = ball.position - player_car.position
        rel_pos = [i - j for i, j in zip(ball.position, player_car.position)]
        # rel_vel = ball.linear_velocity - player_car.linear_velocity
        rel_vel = [i - j for i, j in zip(ball.linear_velocity, player_car.linear_velocity)]

        # obs.extend([
        #     # rel_pos / self.POS_STD,
        #     [i / self.POS_STD for i in rel_pos],
        #     # rel_vel / self.POS_STD,
        #     [i / self.POS_STD for i in rel_vel],
        #     # player_car.position / self.POS_STD,
        #     [i / self.POS_STD for i in player_car.position],
        #     player_car.forward(),
        #     player_car.up(),
        #     # player_car.linear_velocity / self.POS_STD,
        #     [i / self.POS_STD for i in player_car.linear_velocity],
        #     # player_car.angular_velocity / self.ANG_STD,
        #     [i / self.ANG_STD for i in player_car.angular_velocity],
        #     [player.boost_amount,
        #      int(player.on_ground),
        #      int(player.has_flip),
        #      int(player.is_demoed)]])

        obs.extend([i / self.POS_STD for i in rel_pos])
        obs.extend([i / self.POS_STD for i in rel_vel])
        obs.extend([i / self.POS_STD for i in player_car.position])
        obs.extend(player_car.forward())
        obs.extend(player_car.up())
        obs.extend([i / self.POS_STD for i in player_car.linear_velocity])
        obs.extend([i / self.ANG_STD for i in player_car.angular_velocity])
        obs.extend([player.boost_amount, int(player.on_ground), int(player.has_flip), int(player.is_demoed)])

        return player_car
