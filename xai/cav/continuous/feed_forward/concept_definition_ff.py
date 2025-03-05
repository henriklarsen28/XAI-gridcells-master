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
