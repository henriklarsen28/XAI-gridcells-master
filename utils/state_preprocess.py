import torch


def state_preprocess(state: int, device):
        """
        Converts the state to a one-hot encoded tensor,
        that included the position as a one-hot encoding and the orientation as a one-hot encoding.
        """
        field_of_view = state[:-1]
        orientation = int(state[-1])

        field_of_view = torch.tensor(field_of_view, dtype=torch.float32, device=device)

        onehot_vector_orientation = torch.zeros(4, dtype=torch.float32, device=device)
        onehot_vector_orientation[orientation] = 1
        return torch.concat((field_of_view, onehot_vector_orientation))