import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def combine_grad_sam():
    load_path = "./grad_sam/visionary-hill-816_10.pt"
    load_path2 = "./grad_sam/visionary-hill-816_20.pt"

    # merge the two grad_sam files
    grad_sam1 = torch.load(load_path2, map_location=device)
    
    # grad sam is a list containing a dictionary. print the keys of the dictionary within the list
    #print(grad_sam1[0]["tensors"])
    
    count = 0
    for idx, episode in enumerate(grad_sam1):
        print(f"Episode: {idx}")
        step, position, tensors, is_stuck = episode.values()
        print(step, is_stuck)
            # print(tensors)
        count += 1


print("load grad_sam...")
combine_grad_sam()


