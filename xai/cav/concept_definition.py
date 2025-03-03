from collections import deque

from utils import larger_than_less_than


class Concepts:

    def __init__(
            self,
            grid_pos_to_id: dict
        ):
        self.grid_pos_to_id = grid_pos_to_id
        self.datasets = {
            'positive_dataset_wall' : deque(),
            'negative_dataset_wall' : deque(),
            
            'positive_dataset_rotating' : deque(),
            'negative_dataset_rotating' : deque(),

            'positive_dataset_goal' : deque(),
            'negative_dataset_goal' : deque(),

            'positive_dataset_inside_box' : deque(),
            'negative_dataset_inside_box' : deque(),
            
            'grid_observations' : {grid_id: deque() for grid_id in set(self.grid_pos_to_id.values())}
        }

    def clear_datasets(self):
        for dataset in self.datasets:
            # check if the dataset is a dictionary
            if isinstance(self.datasets[dataset], dict):
                for key in self.datasets[dataset]:
                    self.datasets[dataset][key].clear()
            else:
                self.datasets[dataset].clear()

    def looking_at_wall(
        self, observation_sequence: deque, legal_actions: list, action_sequence: deque
    ):
        # Look at the last 2 states, if the agents last states are the same and the agent is not allowed to move forward the agent is stuck in a wall
        # The sequence should be added to the CAV positive dataset

        last_action = action_sequence[-1]

        # print("Last action: ", last_action)
        if len(legal_actions) == 2 and last_action == 0:
            # Save the observation sequence to the positive dataset
            # print("Positive stuck in wall")
            self.datasets['positive_dataset_wall'].append(observation_sequence)
        else:
            self.datasets['negative_dataset_wall'].append(observation_sequence)


    def rotating_stuck(
        self, observation_sequence: deque, action_sequence: deque, position_sequence: deque
    ):

        # The position is the same over the last 12 states, and the agent is rotating in place
        # The sequence should be added to the CAV positive dataset

        position_sequence = list(position_sequence)
        last_12_positions = position_sequence[-12:]
        if len(set(last_12_positions)) == 1:
            # Check if the agent is rotating in place
            action_sequence = list(action_sequence)
            last_12_actions = set(action_sequence[-12:])
            if (1 in last_12_actions or 2 in last_12_actions) and 0 not in last_12_actions:
                # print("Positive rotating stuck")
                self.datasets['positive_dataset_rotating'].append(observation_sequence)
            else:
                self.datasets['negative_dataset_rotating'].append(observation_sequence)


    def goal_in_sight(self, observation_sequence: deque):
        # Check if there is a goal in sight
        # contains a value of 2
        observation = observation_sequence[-1]

        if 2 in observation:
            self.datasets['positive_dataset_goal'].append(observation_sequence)
        else:
            self.datasets['negative_dataset_goal'].append(observation_sequence)

    def inside_box(self, observation_sequence: deque, position: tuple):
        # Coordinates of the box
        coordinates = [(4, 5), (4, 15), (10, 5), (10, 15)]

        # Check if the agent is inside the box
        if larger_than_less_than(
            position[0], coordinates[0][0], coordinates[2][0]
        ) and larger_than_less_than(position[1], coordinates[0][1], coordinates[1][1]):
            # print("Positive inside box")
            # print(position)
            self.datasets['positive_dataset_inside_box'].append(observation_sequence)
        else:
            self.datasets['negative_dataset_inside_box'].append(observation_sequence)


    def stuck_in_wall():
        positive_dataset = deque()

        negative_dataset = deque()

        # Load the dataset

    def in_grid_square(self, observation_sequence: deque, position: tuple):
        '''print("Grid pos to id: ", self.grid_pos_to_id)
        print("Position:", self.grid_pos_to_id.get(position, None))
        print("Position:", position)
        print("grid observation dataset:", self.datasets['grid_observations'])'''
   
        grid_id = self.grid_pos_to_id.get(position, None)
        if grid_id is not None:
            self.datasets['grid_observations'][grid_id].append(observation_sequence)
        else:
            print("Position not in grid")
            # print(position)