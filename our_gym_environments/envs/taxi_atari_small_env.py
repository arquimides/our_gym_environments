from contextlib import closing
from io import StringIO
from os import path
from typing import Optional
import numpy as np
import os
import cv2
from gymnasium import Env, spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]
WINDOW_SIZE = (550, 350)


class TaxiAtariSmallEnv(Env):
    """

    Custom implementation as Atari game of The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    ### Description
    There are four designated locations in the grid world indicated by R(ed),
    G(reen), Y(ellow), and B(lue). When the episode starts, the taxi starts off
    at a random square and the passenger is at a random location. The taxi
    drives to the passenger's location, picks up the passenger, drives to the
    passenger's destination (another one of the four specified locations), and
    then drops off the passenger. Once the passenger is dropped off, the episode ends.

    Map:

        +---------+
        |R: | : :G|
        | : | : : |
        | : : : : |
        | | : | : |
        |Y| : |B: |
        +---------+

    ### Actions
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger

    ### Observations
    There are 500 discrete states since there are 25 taxi positions, 5 possible
    locations of the passenger (including the case when the passenger is in the
    taxi), and 4 destination locations.

    Note that there are 400 states that can actually be reached during an
    episode. The missing states correspond to situations in which the passenger
    is at the same location as their destination, as this typically signals the
    end of an episode. Four additional states can be observed right after a
    successful episodes, when both the passenger and the taxi are at the destination.
    This gives a total of 404 reachable discrete states.

    Each state space is represented by the tuple:
    (taxi_row, taxi_col, passenger_location, destination)

    An observation is an integer that encodes the corresponding state.
    The state tuple can then be decoded with the "decode" method.

    Passenger locations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: in taxi

    Destinations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)

    ### Rewards
    - -1 per step unless other reward is triggered.
    - +20 delivering passenger.
    - -10 executing "pickup" and "drop-off" actions illegally.
    - 0 for enter to the "pickup" or the "drop-off" cell
    - 0 for executing "pickup" correctly.

    ### Arguments

    ```
    gym.make('Taxi-v3')
    ```

    ### Version History
    * v3: Map Correction + Cleaner Domain Description
    * v2: Disallow Taxi start location = goal location, Update Taxi observations in the rollout, Update Taxi reward threshold.
    * v1: Remove (3,2) from locs, add passidx<4 check
    * v0: Initial versions release
    """

    metadata = {"render_modes": ["human", "ansi", "rgb_array", "none", "preprocessed"], "render_fps": 64, "environment_type": ["stochastic", "deterministic"], "reward_type": ["original", "new"]}

    def __init__(self, render_mode=None, env_type="deterministic", reward_type = "original", render_fps=4):

        assert env_type == "deterministic" or env_type in self.metadata["environment_type"]
        self.env_type = env_type
        assert reward_type == "original" or reward_type in self.metadata["reward_type"]
        self.reward_type = reward_type
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_fps = render_fps

        self.desc = np.asarray(MAP, dtype="c")

        # The locations of the possible origin and destination points in the grid world
        self.locs = locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        self.locs_colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]

        # stochastic transition success probabilities
        self.trans_probs = {"south": 0.9, "north": 0.9, "east": 0.9, "west": 0.9, "pick": 0.9, "drop": 0.9}

        self.num_rows = 5
        self.num_columns = 5
        self.pass_destinations = 4
        self.pass_locations = 5  # Same as destinations and one more to indicate "on the taxi"
        self.num_states = self.num_rows * self.num_columns * self.pass_locations * self.pass_destinations

        self.max_row = self.num_rows - 1
        self.max_col = self.num_columns - 1

        self.initial_state_distrib = np.zeros(self.num_states)

        # Atari related, after four crashed the taxi lose a live
        self.max_lives = 15
        self.lives = self.max_lives

        self.s = 0
        self.num_actions = 6
        self.actions = ["south", "north", "east", "west", "pick", "drop"]
        self.state_variables_names = ["wp", "l", "nw"]
        self.state_variables_cardinalities = [2, 3,
                                              15]  # The nw variable card is 15 because there is not posible to have wall in the 4 points. Ex: 1111
        self.reward_variable_name = ["reward"]
        self.reward_variable_categories = [0, 1, 2]  # 0 is good, 1 is normal, 2 is bad
        self.reward_variable_values = [20.0, -1.0, -10.0]  # TODO see if we can use a dict instead
        self.reward_variable_cardinality = 3  # 0-positive, 1-normal, 2-negative

        self.last_action = None
        self.destination = None
        self.origin = None
        self.states = []
        self.relational_states = None
        self.possible_states = []

        self.P = {
            state: {action: [] for action in range(self.num_actions)}
            for state in range(self.num_states)
        }
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                for pass_idx in range(len(locs) + 1):  # +1 for being inside taxi
                    for dest_idx in range(len(locs)):
                        state = self.encode(row, col, pass_idx, dest_idx)
                        if pass_idx != dest_idx:
                            self.initial_state_distrib[state] += 1
                            self.possible_states.append([row,col,pass_idx,dest_idx])
                        for action in range(self.num_actions):
                            # defaults
                            new_row, new_col, new_pass_idx = row, col, pass_idx
                            reward = (
                                self.reward_variable_values[1]
                            )  # default reward when there is no pickup/dropoff
                            done = False
                            taxi_loc = (row, col)

                            if action == 0:  # South
                                if self.desc[row + 2, 2 * col + 1] not in [b"-", b"|", b"+"]:
                                    new_row = min(row + 1, self.max_row)
                            elif action == 1:  # North
                                if self.desc[row, 2 * col + 1] not in [b"-", b"|", b"+"]:
                                    new_row = max(row - 1, 0)
                            elif action == 2:  # East
                                if self.desc[1 + row, 2 * col + 2] not in [b"-", b"|", b"+"]:
                                    new_col = min(col + 1, self.max_col)
                            elif action == 3:  # West
                                if self.desc[1 + row, 2 * col] not in [b"-", b"|", b"+"]:
                                    new_col = max(col - 1, 0)
                            elif action == 4:  # pickup
                                if pass_idx < 4 and taxi_loc == locs[pass_idx]:
                                    new_pass_idx = 4  # passenger is now in the taxi
                                    # reward = 1 #TODO This is critical.
                                else:  # the taxi is not at pass location or the passenger is already in the taxi
                                    reward = self.reward_variable_values[2]
                            elif action == 5:  # drop-off

                                # First assume drop-off at wrong location
                                reward = self.reward_variable_values[2]

                                if (taxi_loc == locs[dest_idx]) and pass_idx == 4:
                                    new_pass_idx = dest_idx
                                    done = True
                                    reward = self.reward_variable_values[0]
                                elif (taxi_loc in locs) and pass_idx == 4:
                                    new_pass_idx = locs.index(taxi_loc)

                            if 0 <= action <= 3 and (row, col) == (new_row, new_col):
                                reward = self.reward_variable_values[2]

                            # if 0 <= action <= 3 and (row,col) != (new_row, new_col) :
                            #     if pass_idx < 4 and new_row == locs[pass_idx][0] and new_col == locs[pass_idx][1]:
                            #         reward = 0.1
                            #         #subgoal = 1
                            #     elif pass_idx == 4 and new_row == locs[dest_idx][0] and new_col == locs[dest_idx][1]:
                            #         reward = 0.1
                            #         #subgoal = 2

                            new_state = self.encode(
                                new_row, new_col, new_pass_idx, dest_idx
                            )

                            if env_type == "deterministic":
                                self.P[state][action].append((1.0, new_state, reward, done))
                            elif env_type == "stochastic":
                                self.P[state][action].append((self.trans_probs[self.actions[action]], new_state, reward, done))
                                self.P[state][action].append((1 - self.trans_probs[self.actions[action]], state, self.reward_variable_values[1], False))

        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.action_space = spaces.Discrete(self.num_actions)

        # I need to chage the original observation space from Integer to Image
        #self.observation_space = spaces.Discrete(self.num_states)
        # Define the observation space
        # self.observation_space = spaces.Box(low=0, high=255, shape=(350, 550, 3), dtype=np.uint8)

        # if self.render_mode == "preprocessed":
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)

        self.info = None
        self.last_action = None
        self.last_reward = 0
        self.total_reward = 0
        self.step_number = 0

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
        self.images_map = {}
        self.maximum_episode_steps = 1000

        # Precargar todas las imagenes procesadas que corresponden a cada esta para no tener que hacer render
        self.images_dict = self.load_preprocessed_images()



    def load_preprocessed_images(self):

        images_dict = {}

        # Get the directory path of the script containing this function
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the full path to the image folder (relative to the script's location)
        image_folder_path = os.path.join(script_dir, "env_images")

        # List all files in the folder
        image_files = os.listdir(image_folder_path)

        for filename in image_files:
            # Extract the state number from the filename (assuming filenames are in the format 'image_XXX.png')
            try:
                state_number = int(filename.split('_')[1].split('.')[0])
            except (ValueError, IndexError):
                print(f"Skipping file {filename} as it doesn't follow the expected naming convention.")
                continue

            # Load the image using OpenCV
            image_path = os.path.join(image_folder_path, filename)
            image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
            # Add a channel dimension to make it (84, 84, 1)
            #image = np.expand_dims(image, axis=-1)

            # Add the image to the dictionary
            images_dict[state_number] = image

        return images_dict


    def _get_info(self):

        state = self.convert_to_relational(self.decode(self.s))

        WP = ["NO", "YES"]
        L = ["On the road", "On origin", "On destination"]
        NW = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]

        return {
            "WP": str(WP[state[0]]),
            "L": str(L[state[1]]),
            "NW": str(NW[state[2]]),
            "last_action": str(self.last_action),
            "last_reward": str(self.last_reward),
            "total_reward": str(self.total_reward),
            "step_number": str(self.step_number)
        }


    def init_relational_states(self):
        states_list = []
        relational_states = set()
        for state_number in range(self.num_states):
            array_state = self.decode(state_number)
            current_state = self.convert_to_relational(array_state)
            states_list.append(current_state)
            relational_states.add(tuple(current_state))
        return states_list, relational_states

    def convert_to_relational(self, array_state):
        wp = 0  # options are (0) without passenger (1) with passenger
        l = 0  # options are (0) in the road (1) in origin (2) in destination
        nw = 0  # options are for 0 to 16 representing the integer corresponding to the binary number
        # of 4 digits indicating the presence of a wall at S-N-E-W

        walls = [0, 0, 0, 0]

        taxi_row = array_state[0]
        taxi_col = array_state[1]
        pass_loc = array_state[2]

        # Check for walls at the four cardinal points
        if self.desc[taxi_row + 2, 2 * taxi_col + 1] in [b"-", b"+", b"|"]:  # South self.desc[row, 2 * col + 1] == b"-"
            walls[0] = 1
        if self.desc[taxi_row, 2 * taxi_col + 1] in [b"-", b"+", b"|"]:  # North self.desc[row, 2 * col + 1] == b"-"
            walls[1] = 1
        if self.desc[1 + taxi_row, 2 * taxi_col + 2] in [b"-", b"+",
                                                         b"|"]:  # East  self.desc[1 + row, 2 * col + 2] == b"|"
            walls[2] = 1
        if self.desc[1 + taxi_row, 2 * taxi_col] in [b"-", b"+", b"|"]:  # West  self.desc[1 + row, 2 * col] == b"|":
            walls[3] = 1

        nw = int(''.join(map(str, walls)), 2)

        # the possible origin and destination are locs = [(0, 0), (0, 4), (4, 0), (4, 3)]

        if pass_loc == 4:  # If the passenger is in the taxi
            wp = 1

        # Check if the taxi is at the origin or at destination
        try:
            index = self.locs.index((taxi_row, taxi_col))
        except:
            index = -1

        if index == self.origin:
            l = 1
        elif index == self.destination:
            l = 2

        return [wp, l, nw]

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        # (5) 5, 5, 4
        i = taxi_row
        i *= 5
        i += taxi_col
        i *= 5
        i += pass_loc
        i *= 4
        i += dest_idx
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return list(reversed(out))

    def step_original(self, a):

        if a < 0 or a > 5:
            return self.s, 0 , False, False,{"prob": 1}

        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.last_action = a
        self.last_reward = r
        self.total_reward += self.last_reward
        self.step_number += 1

        self.info = self._get_info()

        self.render()

        # TODO Substitute the False for Truncated value
        return int(s), r, d, False, {"prob": p}

    def step(self, a):

        if a == 1000:
            self.info = self._get_info()

            if self.render_mode == "human":
                self._render_gui()

            return int(self.s), 0, False, False, {"prob": 1, "description": self.info}

        # First we decode current state
        row, col, pass_idx, dest_idx = self.decode(self.s)

        wp, l, nw = self.convert_to_relational([row, col, pass_idx, dest_idx])

        # By default, the next state is a copy of the current state
        new_row, new_col, new_pass_idx, new_dest_idx = row, col, pass_idx, dest_idx
        reward = (
            self.reward_variable_values[1]  # default reward for most actions
        )
        done = False

        taxi_loc = (row, col)

        prob = np.random.uniform()

        if a == 0:  # South

            if self.env_type == "deterministic":
                if self.desc[row + 2, 2 * col + 1] not in [b"-", b"|", b"+"]:
                    new_row = min(row + 1, self.max_row)
            elif self.env_type == "stochastic":
                if prob < self.trans_probs[self.actions[a]]:
                    if self.desc[row + 2, 2 * col + 1] not in [b"-", b"|", b"+"]:
                        new_row = min(row + 1, self.max_row)

        elif a == 1:  # North

            if self.env_type == "deterministic":
                if self.desc[row, 2 * col + 1] not in [b"-", b"|", b"+"]:
                    new_row = max(row - 1, 0)

            elif self.env_type == "stochastic":
                if prob < self.trans_probs[self.actions[a]]:
                    if self.desc[row, 2 * col + 1] not in [b"-", b"|", b"+"]:
                        new_row = max(row - 1, 0)

        elif a == 2:  # East

            if self.env_type == "deterministic":
                if self.desc[1 + row, 2 * col + 2] not in [b"-", b"|", b"+"]:
                    new_col = min(col + 1, self.max_col)

            elif self.env_type == "stochastic":
                if prob < self.trans_probs[self.actions[a]]:
                    if self.desc[1 + row, 2 * col + 2] not in [b"-", b"|", b"+"]:
                        new_col = min(col + 1, self.max_col)

        elif a == 3:  # West

            if self.env_type == "deterministic":
                if self.desc[1 + row, 2 * col] not in [b"-", b"|", b"+"]:
                    new_col = max(col - 1, 0)
            elif self.env_type == "stochastic":
                if prob < self.trans_probs[self.actions[a]]:
                    if self.desc[1 + row, 2 * col] not in [b"-", b"|", b"+"]:
                        new_col = max(col - 1, 0)

        # Reward for move actions independent of reward_type
        if 0 <= a <= 3 and (row, col) == (new_row, new_col):
            reward = self.reward_variable_values[2]

        elif a == 4:  # pickup

            if self.env_type == "deterministic":
                if pass_idx < 4 and taxi_loc == self.locs[pass_idx]:
                    new_pass_idx = 4  # passenger is now in the taxi

            elif self.env_type == "stochastic":
                if prob < self.trans_probs[self.actions[a]]:
                    if pass_idx < 4 and taxi_loc == self.locs[pass_idx]:
                        new_pass_idx = 4  # passenger is now in the taxi

            if self.reward_type == "original":
                if wp == 1 or l != 1:
                    reward = self.reward_variable_values[2]

            elif self.reward_type == "new":
                # calculate the next relational state
                #new_wp, new_l, new_nw = self.convert_to_relational([new_row, new_col, new_pass_idx, new_dest_idx])

                if wp == 1 or l != 1:
                    reward = self.reward_variable_values[2]

        elif a == 5:  # drop-off

            if self.env_type == "deterministic":
                if (taxi_loc == self.locs[dest_idx]) and pass_idx == 4:
                    new_pass_idx = dest_idx
                    done = True
                elif (taxi_loc in self.locs) and pass_idx == 4:
                    new_pass_idx = self.locs.index(taxi_loc)

            elif self.env_type == "stochastic":
                if prob < self.trans_probs[self.actions[a]]:
                    if (taxi_loc == self.locs[dest_idx]) and pass_idx == 4:
                        new_pass_idx = dest_idx
                        done = True
                    elif (taxi_loc in self.locs) and pass_idx == 4:
                        new_pass_idx = self.locs.index(taxi_loc)

            if self.reward_type == "original":
                if wp == 1 and l == 2:
                    reward = self.reward_variable_values[0]

                elif wp != 1:
                    reward = self.reward_variable_values[2]

            elif self.reward_type == "new":
                # calculate the next relational state
                new_wp, new_l, new_nw = self.convert_to_relational([new_row, new_col, new_pass_idx, new_dest_idx])

                if wp == 1 and l == 2 and new_wp == 0:
                    reward = self.reward_variable_values[0]
                elif wp != 1:
                    reward = self.reward_variable_values[2]

        s = self.encode(new_row, new_col, new_pass_idx, new_dest_idx)

        self.s = s
        self.last_action = a
        self.last_reward = reward
        self.total_reward += self.last_reward
        self.step_number += 1

        self.info = self._get_info()

        obs = self.render()

        truncated = False
        if self.step_number >= self.maximum_episode_steps:
            truncated = True

        # TODO cambiar este metodo para que devuela la observacion como una imagen en lugar del estado entero
        # TODO tambien tengo que devolver el estado entero para poder extraer las variables relacionales
        return obs, reward, done, truncated, {"prob": 1, "description": self.info, "integer_state": int(s), "lives": self.lives}


    def do_task(self, start_state, max_steps, value_function):
            terminated = False
            self.reset(options={'state_index': start_state})
            steps = 0
            episode_reward = 0
            while not terminated and steps < max_steps:
                s = self.s  # Parsing the state to a given index

                # Take the best action on this state according to the value function
                a = np.random.choice(np.where(value_function[s] == np.max(value_function[s]))[0])
                # print(self.actions[a])
                # Take action, observe outcome
                # observation, reward, terminated, truncated, info = self.step(current_state, self.actions[a], environment_type)
                observation, reward, terminated, truncated, info = self.step(a)
                # current_state = next_state
                episode_reward += reward
                steps = steps + 1

            return episode_reward, steps

    def step_relational(self, a):

        # TODO Complete if necesary
        # s = [wp, l]
        WP = ["With Passenger", "Without Passenger"]
        L = ["On road", "On origin", "On destination"]

        # First we decode current state
        state_array = self.decode(self.s)
        # By default, the next state is a copy of the current state
        # new_wp, new_l = wp, l

        reward = (
            -0.1  # default reward for most actions
        )
        done = False

    def random_initial_states(self, episodes):
        resp = []
        for i in range(episodes):
            resp.append(categorical_sample(self.initial_state_distrib, self.np_random))
        return resp

    def random_state_from_relational(self, relational_index):
        s_candidates = []

        relational_state = list(list(self.relational_states)[relational_index])

        for state_number in self.states:
            if self.states[state_number] == relational_state:
                s_candidates.append(state_number)
        return np.random.choice(s_candidates)

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        if options is not None:
            if options['state_type'] == "original":
                row, col, pass_idx, dest_idx = self.possible_states[options['state_index'] % len(self.possible_states)]
                self.s = self.encode(row,col,pass_idx,dest_idx)
            elif options['state_type'] == "relational":
                self.s = self.random_state_from_relational(options['state_index'] % len(self.relational_states))
        else:
            self.s = categorical_sample(self.initial_state_distrib, self.np_random)

        array_state = self.decode(self.s)
        self.origin = array_state[2]
        self.destination = array_state[3]
        if self.origin == 4: # if the passenger is already on the taxi we need to randomly place an origin position distinct of destination
            all_options = list(range(4))
            all_options.remove(self.destination)
            self.origin = np.random.choice(all_options)
        self.states, self.relational_states = self.init_relational_states()
        self.taxi_orientation = 0
        self.last_action = None
        self.last_reward = 0
        self.total_reward = 0
        self.step_number = 0
        self.lives = self.max_lives # Reset the lives counter

        self.info = self._get_info()

        obs = self.render()

        # if not return_info:
        #     return (obs, {})
        # else:
        return (obs, {"prob": 1, "integer_state": int(self.s)})

    def render(self):
        if self.render_mode == "ansi":
            return self._render_text()
        elif self.render_mode in ["human", "rgb_array"]:
            return self._render_gui()
        elif self.render_mode == "preprocessed":
            return self.images_dict[self.s]
        return


    def _render_gui(self):

        state_array = self.decode(self.s)

        try:
            import pygame  # dependency to pygame only if rendering with human
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy_text]`"
            )

        if self.window is None:
            pygame.init()
            pygame.display.set_caption("TaxiSmall")
            if self.render_mode == "human":
                self.window = pygame.display.set_mode(WINDOW_SIZE)
            else:  # "rgb_array"
                self.window = pygame.Surface(WINDOW_SIZE)

        if self.info is not None:
            pygame.display.set_caption("TaxiSmall")
            #print(self.info)

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.taxi_imgs is None:
            file_names = [
                path.join(path.dirname(__file__), "img/cab_front.png"),
                path.join(path.dirname(__file__), "img/cab_rear.png"),
                path.join(path.dirname(__file__), "img/cab_right.png"),
                path.join(path.dirname(__file__), "img/cab_left.png"),
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
            file_name = path.join(path.dirname(__file__), "img/hotel.png")
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
        if self.background_img is None:
            file_name = path.join(path.dirname(__file__), "img/taxi_background.png")
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

        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        if pass_idx < 4:
            self.window.blit(self.passenger_img, self.get_surf_loc(self.locs[pass_idx]))

        if self.last_action in [0, 1, 2, 3]:
            self.taxi_orientation = self.last_action
        dest_loc = self.get_surf_loc(self.locs[dest_idx])
        taxi_location = self.get_surf_loc((taxi_row, taxi_col))

        if dest_loc[1] <= taxi_location[1]:
            self.window.blit(
                self.destination_img,
                (dest_loc[0], dest_loc[1] - self.cell_size[1] // 2),
            )
            self.window.blit(self.taxi_imgs[self.taxi_orientation], taxi_location)
        else:  # change blit order for overlapping appearance
            self.window.blit(self.taxi_imgs[self.taxi_orientation], taxi_location)
            self.window.blit(
                self.destination_img,
                (dest_loc[0], dest_loc[1] - self.cell_size[1] // 2),
            )

        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.render_fps)
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
        if self.last_action is not None:
            outfile.write(
                f"  ({['South', 'North', 'East', 'West', 'Pickup', 'Dropoff'][self.last_action]})\n"
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


