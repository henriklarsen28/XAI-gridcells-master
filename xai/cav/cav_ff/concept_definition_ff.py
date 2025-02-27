from collections import deque

class Concepts:

    def __init__(
            self,
            grid_pos_to_id: dict
        ):
        self.grid_pos_to_id = grid_pos_to_id
        self.datasets = {
            'positive_dataset_looking_at_wall' : deque(),
            'negative_dataset_looking_at_wall' : deque(),
            
            'positive_dataset_rotating' : deque(),
            'negative_dataset_rotating' : deque(),
            
            'positive_next_to_wall' : deque(),
            'negative_next_to_wall' : deque(),

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

    def positive_looking_at_wall(self, sequence: deque, legal_actions: list, action_sequence: deque):
        # Look at the last 2 states, if the agents last states are the same and the agent is not allowed to move forward the agent is stuck in a wall
        # The sequence should be added to the CAV positive dataset

        last_action = action_sequence[-1]

        #print("Last action: ", last_action)
        if len(legal_actions) == 2 and last_action == 0:
            # Save the observation sequence to the positive dataset
            #print("Positive stuck in wall")
            self.datasets['positive_dataset_looking_at_wall'].append(sequence)
        else:
            self.datasets['negative_dataset_looking_at_wall'].append(sequence)

        return None


    def positive_rotating_stuck(
        self, sequence: deque, action_sequence: deque, position_sequence: deque
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
                #print("Positive rotating stuck")
                self.datasets['positive_dataset_rotating'].append(sequence)
            else:
                self.datasets['negative_dataset_rotating'].append(sequence)

    def positive_next_to_wall(self, observation, position:tuple, matrix_width:int):
        # Check if the there is a wall to the left or right of the agent
        # Positive:
        # 0 0 0 1 0 0 0
        # 0 1 1 1 0 0 0
        # 1 1 1 1 0 0 0

        # Positive: Looking straight at a wall
        # 0 0 0 1 0 0 0
        # 0 0 0 0 0 0 0
        # 0 0 0 0 0 0 0


        # Negative:
        # 0 0 0 1 0 0 0
        # 0 0 1 1 1 0 0
        # 0 0 1 1 1 0 0
        
        
        left_wall = False
        right_wall = False

        # Unflatten the observation
        observation_unflattened = [observation[i:i+matrix_width] for i in range(0, len(observation), matrix_width)]
        middle_index = len(observation_unflattened[0]) // 2
        observation_unflattened = observation_unflattened[:-1]
        # Skip the first row because that is the agent
        observation_unflattened = observation_unflattened[1:]
        # Only need to check the middle column +- 1 column
        for row in observation_unflattened:
            if row[middle_index] == 0:
                break

            if row[middle_index - 1] == 0:
                left_wall = True
                break
            if row[middle_index + 1] == 0:
                right_wall = True
                break
        
            

        if left_wall or right_wall:
            self.datasets['positive_next_to_wall'].append(observation)
        else:
            self.datasets['negative_next_to_wall'].append(observation)
 
        
    def positive_goal_in_sight(self, observation):
        # Check if there is a goal in sight
        # contains a value of 2

        if 2 in observation:
            self.datasets['positive_dataset_goal'].append(observation)
        else:
            self.datasets['negative_dataset_goal'].append(observation)

    def in_grid_square(self, observation, position: tuple):
        '''print("Grid pos to id: ", self.grid_pos_to_id)
        print("Position:", self.grid_pos_to_id.get(position, None))
        print("Position:", position)
        print("grid observation dataset:", self.datasets['grid_observations'])'''
   
        grid_id = self.grid_pos_to_id.get(position, None)
        if grid_id is not None:
            self.datasets['grid_observations'][grid_id].append(observation)
        else:
            print("Position not in grid")
            # print(position)
