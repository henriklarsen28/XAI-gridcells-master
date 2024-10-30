import torch
import pprint
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def combine_grad_sam():
    load_path = "./grad_sam/map_open_doors_90_degrees/visionary-hill-816_10.pt"
    load_path2 = "./grad_sam/map_open_doors_90_degrees/visionary-hill-816_20.pt"

    # merge the two grad_sam files
    grad_sam1 = torch.load(load_path, map_location=device)
    
    # grad sam is a list containing a dictionary. print the keys of the dictionary within the list
    #print(grad_sam1[0]["tensors"])

    # print(grad_sam1[0]['tensors'][0].shape)
    
    ''' count = 0
    for idx, episode in enumerate(grad_sam1):
        
        print(f"Episode: {idx}")
        for step in episode:
            print(step.values())
            # print(tensors)
        count += 1'''
    
    with open('grad_sam_structure.txt', 'w') as f:
        f.write(pprint.pformat(grad_sam1))
    # pprint.pprint(grad_sam1)

    # print(json.dumps(grad_sam1, indent=4))

   

combine_grad_sam()


