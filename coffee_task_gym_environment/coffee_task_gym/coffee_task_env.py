from contextlib import closing
from io import StringIO
from os import path
from typing import Optional

import numpy as np

from gym import Env, spaces, utils
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled

MAP = [
    "+---------+",
    "| : : : : |",
    "|R:G: : : |",
    "| : : : : |",
    "| : : : : |",
    "| : : :Y:B|",
    "+---------+",
]
WINDOW_SIZE = (800, 600)


class CoffeeTaskEnv(Env):
    """

    The Coffee Task Problem
    from "Exploiting structure in policy construction"
    by Craig Boutlier

    ### Description

    An office robot is ask to go to a coffee shop, buy coffee, and return to deliver it to a user in her office.
    On the way it could be raining, so the robot will get wet unless it has taken an umbrella (available at the office)
    before leaving. When an episode starts the robot must be on any non-terminal state.
    Once the user has the coffee, the episode ends.

    This is the Map used to place the House and Coffee shop, just for render purposes:

        +---------+
        | : : : : |
        |R:G: : : |
        | : : : : |
        | : : : : |
        | : : :Y:B|
        +---------+

    ### Actions
    There are 4 discrete deterministic actions:
    - 0: GO changing the robot's location and the robot can get wet if it rains and it does not have an umbrella
    - 1: GU causes it to hold an umbrella if it is in the office
    - 2: BC causes the robot to hold coffee if it is in the coffee shop
    - 3: DC causes the user to hold coffee if the robot has coffee and is in the office

    ### Observations
    A state is described by six binary state variables:
    - SL, the location of the robot (office or coffee shop);
    - SU, whether the robot has an umbrella;
    - SR, whether it is raining;
    - SW, whether the robot is wet;
    - SC, whether the robot has coffee;
    - SH, whether the user has coffee.

    There are 64 discrete states. Each state is distinct combination of the state variables.
    Some states are not reachable, for example those where both, the user and the robot has a coffee.

    Each state space is represented by the tuple:
    (sl, su, sr, sw, sc, sh)

    An observation is an integer that encodes the corresponding state.
    The state tuple can then be decoded with the "decode" method.

    ### Rewards

    The robot gets a reward of $0.9$ whenever the user has coffee plus a reward of $0.1$ whenever it is dry
    In addition, it receives a positive reward of $0.05$ on each sub-goal (being in the coffee shop, buying the coffee
    and returning to the office).
    A penalty of $-1.0$ is given if the robot does not take the umbrella, and it rains,
    and $-0.1$ is given in all other cases.

    ### Transition Probabilities
    If the environment_type is stochastic, there are some variables controlling the transitions probabilities:

    go_change_location_probability = 0.9
    go_get_wet_if_rain_and_not_umbrella_probability = 0.9
    gu_ok_probability = 0.9
    bc_ok_probability = 0.8
    dc_user_have_and_robot_not_probability = 0.8
    dc_in_coffee_shop_probability = 0.9
    rain_probability = 0.3
    rain_stop_probability = 0.3
    max_rain = 25
    rain_time = 0

    ### Arguments

    ```
    environment_type: "deterministic" or "stochastic"
    gym.make('CoffeeTaskEnv-v0', environment_type)
    ```

    ### Version History
    * v0: Initial versions release
    """

    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 4, "environment_type": ["stochastic", "deterministic"]}

    def __init__(self, env_type = "stochastic"):

        self.env_type = env_type
        self.desc = np.asarray(MAP, dtype="c")

        self.locs = locs = [(1, 0), (1, 1), (4, 3), (4, 4)]
        self.locs_colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]

        # stochastic transition probabilities
        self.go_change_location_probability = 0.9
        self.go_get_wet_if_rain_and_not_umbrella_probability = 0.9
        self.gu_ok_probability = 0.9
        self.bc_ok_probability = 0.8
        self.dc_user_have_and_robot_not_probability = 0.8
        self.dc_in_coffee_shop_probability = 0.9
        self.rain_probability = 0.3
        self.rain_stop_probability = 0.3
        self.max_rain = 25
        self.rain_time = 0

        self.num_states = 64
        num_rows = 5
        num_columns = 5
        max_row = num_rows - 1
        max_col = num_columns - 1

        # Initially, all states are possible
        self.initial_state_distrib = np.zeros(self.num_states)
        # But, actually, we need to remove all states ending in "1", meaning that the user has the coffee, because
        # they are terminal states. So, all the possible states must end in "0". If we take into account the binary
        # representation of the state we can say that all possible states are "even".
        # The next line do the trick adding 1 to all even positions and keep 0 at odd positions
        self.initial_state_distrib[::2] += 1
        # Finally, we normalize the distribution
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.s = 0
        num_actions = 4
        self.actions = ["GO", "GU", "BC", "DC"]

        self.P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(self.num_states)
        }

        # for sl in range(2):
        #     for su in range(2):
        #         for sr in range(2):
        #             for sw in range(2):
        #                 for sc in range(2):
        #                     for sh in range(2):
        #                         state = self.encode(sl, su, sr, sw, sc, sh)
        #                         self.initial_state_distrib[state] += 1
        #                         for action in range(num_actions):
        #                             # by default the next state is a copy of the current state
        #                             new_sl, new_su, new_sr, new_sw, new_sc, new_sh = sl, su, sr, sw, sc, sh
        #                             reward = (
        #                                 -0.1 # default reward for most actions
        #                             )
        #                             done = False
        #
        #                             if action == 0:  #GO causing its location to change and the robot to get wet if it is raining and it does not have an umbrella
        #                                 if env_type == "deterministic":
        #                                     new_sl = 1 - sl  # this just change the value form 0 to 1 or vice
        #                                     if sl == 0 and su == 0 and sr == 1 and sw == 0:
        #                                         new_sw = 1
        #                                         reward = -1.0
        #
        #                                 elif env_type == "stochastic":
        #                                     if np.random.uniform() < self.go_change_location_probability:
        #                                         new_sl = 1 - sl  # this just change the value form 0 to 1 or vice
        #
        #                                         if np.random.uniform() < self.go_get_wet_if_rain_and_not_umbrella_probability:
        #                                             if sl == 0 and su == 0 and sr == 1 and sw == 0:
        #                                                 new_sw = 1
        #                                                 reward = -1.0
        #
        #                                 # Immediate reward for GO.
        #                                 # If robot is not holding a coffee and it is at home and now it is at the coffee shop, r = 0.05
        #                                 if sl == 0 and sc == 0 and new_sl == 1:
        #                                     reward = 0.05
        #                                 # If robot is holding a coffee and it is at coffee shop and now it is at home, r = 0.05
        #                                 elif sl == 1 and sc == 1 and new_sl == 0:
        #                                     reward = 0.05
        #
        #                             elif action == 1:  #GU causing it to hold an umbrella if it is in the office
        #                                 if env_type == "deterministic":
        #                                     if sl == 0:
        #                                         new_su = 1
        #
        #                                 elif env_type == "stochastic":
        #                                     if np.random.uniform() < self.gu_ok_probability:
        #                                         if sl == 0:
        #                                             new_su = 1
        #
        #                                 # Reward for GU.
        #                                 # If robot is at home, not wearing and umbrella and it is raining, r = 0.05
        #                                 if sl == 0 and su == 0 and sr == 1:
        #                                     reward = 0.05
        #
        #                             elif action == 2:  # BC causing it to hold coffee if it is in the coffee shop
        #                                 if env_type == "deterministic":
        #                                     if sl == 1:
        #                                         new_sc = 1
        #
        #                                 elif env_type == "stochastic":
        #                                     if np.random.uniform() < self.bc_ok_probability:
        #                                         if sl == 1:
        #                                             new_sc = 1
        #
        #                                 # Reward for BC.
        #                                 # If robot is not holding a coffee and it is at coffee shop and now it is holding a coffee, r = 0.1
        #                                 if sl == 1 and sc == 0 and new_sc == 1:
        #                                     reward = 0.05
        #
        #                             elif action == 3:  #DC causing the user to hold coffee and the robot to not hold a coffee if the robot has coffee and is in the office
        #                                 if env_type == "deterministic":
        #                                     new_sc = 0  # and the robot has not
        #                                     if sc == 1 and sl == 0:
        #                                         new_sh = 1  # now the user has a coffee
        #                                         done = True
        #                                 elif env_type == "stochastic":
        #                                     prob = np.random.uniform()
        #
        #                                     if sc == 1 and sl == 0:
        #                                         if prob < self.dc_user_have_and_robot_not_probability:
        #                                             new_sh = 1  # now the user has a coffee
        #                                             new_sc = 0  # and the robot has not
        #
        #                                         elif self.dc_user_have_and_robot_not_probability <= prob < 0.9:  # the robot just dropped the coffee
        #                                             new_sc = 0
        #
        #                                     elif sc == 1 and sl == 1:
        #                                         if prob < self.dc_in_coffee_shop_probability:
        #                                             new_sc = 0
        #
        #                                 # Obtaining reward for DC.
        #                                 # The robot gets a reward of 0.9 whenever the user has coffee plus a reward of 0.1 whenever it is dry
        #                                 if new_sh == 1 and new_sw == 0:
        #                                     r = 1
        #                                     done = True
        #                                 elif new_sh == 1 and new_sw == 1:
        #                                     r = 0.9
        #                                     done = True
        #
        #                             # # Eventually it is raining
        #                             if sr:
        #                                 # if so, it can continue for next step or just stop raining at any time or at max_rain
        #                                 self.rain_time = self.rain_time + 1
        #                                 if np.random.uniform() < self.rain_stop_probability or self.rain_time == self.max_rain:
        #                                     new_sr = 0
        #                                     self.rain_time = 0
        #                             else:
        #                                 # if not raining there is a probability to start to rain in next episode
        #                                 if np.random.uniform() < self.rain_probability:
        #                                     new_sr = 1


        # for row in range(num_rows):
        #     for col in range(num_columns):
        #         for pass_idx in range(len(locs) + 1):  # +1 for being inside taxi
        #             for dest_idx in range(len(locs)):
        #                 state = self.encode(row, col, pass_idx, dest_idx)
        #                 if pass_idx < 4 and pass_idx != dest_idx:
        #                     self.initial_state_distrib[state] += 1
        #                 for action in range(num_actions):
        #                     # defaults
        #                     new_row, new_col, new_pass_idx = row, col, pass_idx
        #                     reward = (
        #                         -1
        #                     )  # default reward when there is no pickup/dropoff
        #                     done = False
        #                     taxi_loc = (row, col)
        #
        #                     if action == 0:
        #                         new_row = min(row + 1, max_row)
        #                     elif action == 1:
        #                         new_row = max(row - 1, 0)
        #                     if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
        #                         new_col = min(col + 1, max_col)
        #                     elif action == 3 and self.desc[1 + row, 2 * col] == b":":
        #                         new_col = max(col - 1, 0)
        #                     elif action == 4:  # pickup
        #                         if pass_idx < 4 and taxi_loc == locs[pass_idx]:
        #                             new_pass_idx = 4 # passenger is now in the taxi
        #                         else:  # passenger not at location
        #                             reward = -10
        #                     elif action == 5:  # dropoff
        #                         if (taxi_loc == locs[dest_idx]) and pass_idx == 4:
        #                             new_pass_idx = dest_idx
        #                             done = True
        #                             reward = 20
        #                         elif (taxi_loc in locs) and pass_idx == 4:
        #                             new_pass_idx = locs.index(taxi_loc)
        #                         else:  # dropoff at wrong location
        #                             reward = -10
        #
        #                     # if 0 <= action <= 3 and (row,col) != (new_row, new_col) :
        #                     #     if pass_idx < 4 and new_row == locs[pass_idx][0] and new_col == locs[pass_idx][1]:
        #                     #         reward = 0.1
        #                     #         #subgoal = 1
        #                     #     elif pass_idx == 4 and new_row == locs[dest_idx][0] and new_col == locs[dest_idx][1]:
        #                     #         reward = 0.1
        #                     #         #subgoal = 2
        #
        #                     new_state = self.encode(
        #                         new_row, new_col, new_pass_idx, dest_idx
        #                     )
        #                     self.P[state][action].append((1.0, new_state, reward, done))

        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(self.num_states)

        # pygame utils
        self.window = None
        self.clock = None
        self.cell_size = (
            WINDOW_SIZE[0] / self.desc.shape[1],
            WINDOW_SIZE[1] / self.desc.shape[0],
        )
        self.taxi_imgs = None
        self.taxi_orientation = 0
        self.passenger_img = None
        self.destination_img = None
        self.median_horiz = None
        self.median_vert = None
        self.background_img = None

    def encode(self, sl, su, sr, sw, sc, sh):
        # the state encode is just the binary representation of the state variable list from left to right
        i = 32*sl
        i += 16*su
        i += 8*sr
        i += 4*sw
        i += 2*sc
        i += sh
        return i

    def decode(self, i):

        binary = format(i, "b").zfill(6)
        return [int(binary[0]), int(binary[1]), int(binary[2]), int(binary[3]), int(binary[4]), int(binary[5])]

    def step_original(self, a):

        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a

        return (int(s), r, d, {"prob": p})

    def step(self, a):

        # s = [sl, su, sr, sw, sc, sh]
        SL = ["Office", "CoffeeShop"]
        SU = ["have not an umbrella", "has a umbrella"]
        SR = ["It is NOT raining", "It IS raining"]
        SW = ["It is NOT wet", "It IS wet"]
        SC = ["It is NOT holding a coffee", "It IS holding a coffee"]
        SH = ["The user is NOT holding a coffee", "The user IS holding a coffee"]

        # First we decode current state
        sl, su, sr, sw, sc, sh = self.decode(self.s)

        # By default, the next state is a copy of the current state
        new_sl, new_su, new_sr, new_sw, new_sc, new_sh = sl, su, sr, sw, sc, sh
        reward = (
            -0.1  # default reward for most actions
        )
        done = False

        if a == 0:  # GO causing its location to change and the robot to get wet if it is raining and it does not have an umbrella
            if self.env_type == "deterministic":
                new_sl = 1 - sl  # this just change the value form 0 to 1 or vice
                if sl == 0 and su == 0 and sr == 1 and sw == 0:
                    new_sw = 1
                    reward = -1.0

            elif self.env_type == "stochastic":
                if np.random.uniform() < self.go_change_location_probability:
                    new_sl = 1 - sl  # this just change the value form 0 to 1 or vice

                    if np.random.uniform() < self.go_get_wet_if_rain_and_not_umbrella_probability:
                        if sl == 0 and su == 0 and sr == 1 and sw == 0:
                            new_sw = 1
                            reward = -1.0

            # Immediate reward for GO.
            # If robot is not holding a coffee and it is at home and now it is at the coffee shop, r = 0.05
            if sl == 0 and sc == 0 and new_sl == 1:
                reward = 0.05
            # If robot is holding a coffee and it is at coffee shop and now it is at home, r = 0.05
            elif sl == 1 and sc == 1 and new_sl == 0:
                reward = 0.05

        elif a == 1:  # GU causing it to hold an umbrella if it is in the office
            if self.env_type == "deterministic":
                if sl == 0:
                    new_su = 1

            elif self.env_type == "stochastic":
                if np.random.uniform() < self.gu_ok_probability:
                    if sl == 0:
                        new_su = 1

            # Reward for GU.
            # If robot is at home, not wearing and umbrella and it is raining, r = 0.05
            if sl == 0 and su == 0 and sr == 1:
                reward = 0.05

        elif a == 2:  # BC causing it to hold coffee if it is in the coffee shop
            if self.env_type == "deterministic":
                if sl == 1:
                    new_sc = 1

            elif self.env_type == "stochastic":
                if np.random.uniform() < self.bc_ok_probability:
                    if sl == 1:
                        new_sc = 1

            # Reward for BC.
            # If robot is not holding a coffee and it is at coffee shop and now it is holding a coffee, r = 0.1
            if sl == 1 and sc == 0 and new_sc == 1:
                reward = 0.05

        elif a == 3:  # DC causing the user to hold coffee and the robot to not hold a coffee if the robot has coffee and is in the office
            if self.env_type == "deterministic":
                new_sc = 0  # and the robot has not
                if sc == 1 and sl == 0:
                    new_sh = 1  # now the user has a coffee
                    done = True
            elif self.env_type == "stochastic":
                prob = np.random.uniform()

                if sc == 1 and sl == 0:
                    if prob < self.dc_user_have_and_robot_not_probability:
                        new_sh = 1  # now the user has a coffee
                        new_sc = 0  # and the robot has not

                    elif self.dc_user_have_and_robot_not_probability <= prob < 0.9:  # the robot just dropped the coffee
                        new_sc = 0

                elif sc == 1 and sl == 1:
                    if prob < self.dc_in_coffee_shop_probability:
                        new_sc = 0

            # Obtaining reward for DC.
            # The robot gets a reward of 0.9 whenever the user has coffee plus a reward of 0.1 whenever it is dry
            if new_sh == 1 and new_sw == 0:
                reward = 1
                done = True
            elif new_sh == 1 and new_sw == 1:
                reward = 0.9
                done = True

        # # Eventually it is raining
        if sr:
            # if so, it can continue for next step or just stop raining at any time or at max_rain
            self.rain_time = self.rain_time + 1
            if np.random.uniform() < self.rain_stop_probability or self.rain_time == self.max_rain:
                new_sr = 0
                self.rain_time = 0
        else:
            # if not raining there is a probability to start to rain in next episode
            if np.random.uniform() < self.rain_probability:
                new_sr = 1

        s = self.encode(new_sl, new_su, new_sr, new_sw, new_sc, new_sh)

        self.s = s
        self.lastaction = a

        info = "The robot take the action {}. Now it is at {} and {}. It {}. {}. {}. The user {}".format(self.actions[a], SL[
            new_sl], SR[new_sr], SU[new_su], SW[new_sw], SC[new_sc], SH[new_sh])

        #print (info)

        return (int(s), reward, done, {"prob": 1, "description": info})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
        state_index = None
    ):
        if state_index is not None:
            super().reset(seed=seed)
            self.s = state_index
            self.lastaction = None
            self.taxi_orientation = 0
        else:
            super().reset(seed=seed)
            self.s = categorical_sample(self.initial_state_distrib, self.np_random)
            self.lastaction = None
            self.taxi_orientation = 0
        if not return_info:
            return int(self.s)
        else:
            return int(self.s), {"prob": 1}

    def render(self, mode="human"):
        if mode == "ansi":
            return self._render_text()
        else:
            return self._render_gui(mode)

    def _render_gui(self, mode):

        sl, su, sr, sw, sc, sh = self.decode(self.s)

        try:
            import pygame  # dependency to pygame only if rendering with human
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install pygame`"
            )

        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Coffee Task")
            if mode == "human":
                self.window = pygame.display.set_mode(WINDOW_SIZE)
            else:  # "rgb_array"
                self.window = pygame.Surface(WINDOW_SIZE)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.taxi_imgs is None:
            file_names = [
                path.join(path.dirname(__file__), "img/robot_front.png"),
                path.join(path.dirname(__file__), "img/robot_with_coffee.png"),
                path.join(path.dirname(__file__), "img/robot_with_umbrella.png"),
                path.join(path.dirname(__file__), "img/robot_with_umbrella_and_coffee.png"),
            ]
            self.taxi_imgs = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.passenger_img is None:
            file_name = path.join(path.dirname(__file__), "img/passenger.png")
            self.passenger_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.destination_img is None:
            file_name = path.join(path.dirname(__file__), "img/coffee-machine.png")
            self.destination_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
            self.destination_img.set_alpha(170)
        if self.median_horiz is None:
            file_names = [
                path.join(path.dirname(__file__), "img/gridworld_median_left.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_horiz.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_right.png"),
            ]
            self.median_horiz = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.median_vert is None:
            file_names = [
                path.join(path.dirname(__file__), "img/gridworld_median_top.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_vert.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_bottom.png"),
            ]
            self.median_vert = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        # if self.background_img is None:
        if sr == 0:
            file_name = path.join(path.dirname(__file__), "img/taxi_background.png")
        elif sr == 1:
            file_name = path.join(path.dirname(__file__), "img/background_raining.png")
        self.background_img = pygame.transform.scale(
            pygame.image.load(file_name), self.cell_size
        )

        desc = self.desc

        for y in range(0, desc.shape[0]):
            for x in range(0, desc.shape[1]):
                cell = (x * self.cell_size[0], y * self.cell_size[1])
                self.window.blit(self.background_img, cell)
                if desc[y][x] == b"|" and (y == 0 or desc[y - 1][x] != b"|"):
                    self.window.blit(self.median_vert[0], cell)
                elif desc[y][x] == b"|" and (
                    y == desc.shape[0] - 1 or desc[y + 1][x] != b"|"
                ):
                    self.window.blit(self.median_vert[2], cell)
                elif desc[y][x] == b"|":
                    self.window.blit(self.median_vert[1], cell)
                elif desc[y][x] == b"-" and (x == 0 or desc[y][x - 1] != b"-"):
                    self.window.blit(self.median_horiz[0], cell)
                elif desc[y][x] == b"-" and (
                    x == desc.shape[1] - 1 or desc[y][x + 1] != b"-"
                ):
                    self.window.blit(self.median_horiz[2], cell)
                elif desc[y][x] == b"-":
                    self.window.blit(self.median_horiz[1], cell)

        for cell, color in zip(self.locs, self.locs_colors):
            color_cell = pygame.Surface(self.cell_size)
            color_cell.set_alpha(128)
            color_cell.fill(color)
            loc = self.get_surf_loc(cell)
            self.window.blit(color_cell, (loc[0], loc[1] + 10))

        # if pass_idx < 4:
        self.window.blit(self.passenger_img, self.get_surf_loc(self.locs[0]))

        # if self.lastaction in [0, 1, 2, 3]:
        #     self.taxi_orientation = self.lastaction
        dest_loc = self.get_surf_loc(self.locs[3])

        if sl == 0:
            taxi_location = self.get_surf_loc((1, 1))
        elif sl == 1:
            taxi_location = self.get_surf_loc((4, 3))

        if sc == 0 and su == 0: # without coffee and without umbrella
            self.window.blit(self.taxi_imgs[0], taxi_location)
        elif sc == 0 and su == 1: # without coffee and wit umbrella
            self.window.blit(self.taxi_imgs[2], taxi_location)
        elif sc == 1 and su == 0: # with coffee and without umbrella
            self.window.blit(self.taxi_imgs[1], taxi_location)
        elif sc == 1 and su == 1:  # with coffee and with umbrella
            self.window.blit(self.taxi_imgs[3], taxi_location)

        if dest_loc[1] <= taxi_location[1]:
            self.window.blit(
                self.destination_img,
                (dest_loc[0], dest_loc[1] - self.cell_size[1] // 2),
            )
            #self.window.blit(self.taxi_imgs[self.taxi_orientation], taxi_location)
        else:  # change blit order for overlapping appearance
            #self.window.blit(self.taxi_imgs[self.taxi_orientation], taxi_location)
            self.window.blit(
                self.destination_img,
                (dest_loc[0], dest_loc[1] - self.cell_size[1] // 2),
            )

        if mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def get_surf_loc(self, map_loc):
        return (map_loc[1] * 2 + 1) * self.cell_size[0], (
            map_loc[0] + 1
        ) * self.cell_size[1]

    def _render_text(self):
        desc = self.desc.copy().tolist()
        outfile = StringIO()

        out = [[c.decode("utf-8") for c in line] for line in desc]
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        def ul(x):
            return "_" if x == " " else x

        if pass_idx < 4:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], "yellow", highlight=True
            )
            pi, pj = self.locs[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(
                out[1 + pi][2 * pj + 1], "blue", bold=True
            )
        else:  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), "green", highlight=True
            )

        di, dj = self.locs[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], "magenta")
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write(
                f"  ({['South', 'North', 'East', 'West', 'Pickup', 'Dropoff'][self.lastaction]})\n"
            )
        else:
            outfile.write("\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
