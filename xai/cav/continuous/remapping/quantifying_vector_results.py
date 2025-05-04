from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


from vector_field import VectorField, create_vector_field

# for measuring rotation pattern in vector field
def rotation_vectors(vector_field: VectorField, grid_length: int):
    
    center_point = (-grid_length / 2, grid_length / 2)
    print("Center point", center_point)
    length_dist = []

    for coordinate, vector in vector_field.vectors.items():
        
        vector_length = np.linalg.norm(vector)

        center_of_vector = (
            coordinate[0] + vector[0] / 2,
            coordinate[1] + vector[1] / 2,
        )
        dist_from_center = np.linalg.norm(np.array(center_of_vector) - np.array(center_point))
        dicti = {
            "length": vector_length,
            "dist_from_center": dist_from_center,
        }

        length_dist.append(dicti)

    plt.figure(figsize=(10, 5))

    # Linear regression
    X = np.array([d["dist_from_center"] for d in length_dist]).reshape(-1, 1)
    y = np.array([d["length"] for d in length_dist])

    model = SVR(kernel="linear")
    #model = LinearRegression()
    model.fit(X, y)

    # Draw the regression line

    x_line = np.linspace(0, 5, 10).reshape(-1, 1)
    y_pred = model.predict(x_line)
    plt.plot(x_line, y_pred, color="red", label="Regression Line")

    
    plt.scatter(
        [d["dist_from_center"] for d in length_dist],
        [d["length"] for d in length_dist],
        alpha=0.5,
    )
    plt.title("Length vs Distance from Center")
    plt.xlabel("Distance from Center")
    plt.ylabel("Length")
    plt.xlim(0, 4.5)
    plt.ylim(0, 8.5)
    plt.grid()

    plt.show()


def nearest_neighbour(vector: np.ndarray, target_vectors, k: int = 3) -> np.ndarray:
    # Compute Euclidean distances to all vectors in V
    distances = np.linalg.norm(target_vectors - vector, axis=1)

    # Get indices of the 3 smallest distances
    nearest_indices = np.argsort(distances)[:k]

    # Get the nearest vectors
    nearest_vectors = target_vectors[nearest_indices]

    return nearest_vectors


def compare_to_target(vector_field: VectorField, target_vector_field: VectorField, grid_length: int):
    
    # match the dictionary keys to ensure they are the same and sort the dictionaries, keep the values
    vector_field_dict = dict(sorted(vector_field.vectors.items()))
    target_vector_field_dict = dict(sorted(target_vector_field.vectors.items()))
    # check if the keys are the same

    print("Vector field keys", vector_field_dict.keys())
    print("Target vector field keys", target_vector_field_dict.keys())


    # drop the keys that are not in the target vector field
    #vector_field_dict = {k: vector_field_dict[k] for k in vector_field_dict.keys() if k in target_vector_field_dict}
    # second method is to find the nearest neighbour


    #if vector_field_dict.keys() != target_vector_field_dict.keys():
    #    raise ValueError("The keys of the two dictionaries are not the same.")
    

    # Calculate the cosine similarity between the two vector fields
    vector_field_values = np.array(list(vector_field_dict.values()))
    target_vector_field_values = np.array(list(target_vector_field_dict.values()))
    
    for vector in vector_field_values:

        nearest_target_vectors = nearest_neighbour(vector, target_vector_field_values, k=3)

        # Calculate the cos sim for all the nearest target vectors
        cos_sim_vectors = cosine_similarity(vector.reshape(1, -1), nearest_target_vectors)
        cos_sim = np.max(cos_sim_vectors, axis=1)
        #select the max cosine similarity
        target_vector_field_index = np.argmax(cos_sim)
        target_vector = nearest_target_vectors[target_vector_field_index]


        #cos_sim = cosine_similarity(vector.reshape(1, -1), target_vector.reshape(1, -1))

        
        #print(f"Vector: {vector}")
        print(f"Target vector: {target_vector}")
        print(f"Cosine similarity: {cos_sim}")
        print(f"Difference in length: {np.linalg.norm(vector) - np.linalg.norm(target_vector)}")

def compare_vector_direction(vector_field: VectorField, vertical: bool = False, range: int = 45):

    """
    See how many of the vectors are in the same direction as expected, within a certain range.
    """

    count = 0

    if vertical:
        expected_range = (270 - range/2, 270 + range/2)

    else:
        expected_range = (0 + range/2, 360 - range/2)

    print(expected_range)

    for coordinate, vector in vector_field.vectors.items():
        angle = np.degrees(np.arctan2(vector[0], vector[1]))
        if angle < 0:
            angle += 360

        if  angle <= expected_range[0] or angle >= expected_range[1]:
            print("Angle", angle)
            count += 1
        else:
            print("vector", vector, "Angle", angle)
            



    print(f"Number of vectors in the expected direction: {count}", "out of", len(vector_field.vectors))
    print("Percentage of vectors in the expected direction: ", count / len(vector_field.vectors) * 100, "%")

    # Plot the vectors with the expected direction
    #plt.figure(figsize=(10, 5))
    angle_rad = np.deg2rad(expected_range[0])

    # Length of vector (e.g., 1 unit)
    length = 100
    x = length * np.cos(angle_rad)
    y = length * np.sin(angle_rad)

    plt.plot([0, x], [0, y], 'r--', linewidth=2)  # red dashed line
    angle_rad = np.deg2rad(expected_range[1])
    x = length * np.cos(angle_rad)
    y = length * np.sin(angle_rad)
    plt.plot([0, x], [0, y], 'r--', linewidth=2)  # red dashed line    
    vector_field.plot_field_centered()
    

def main():
    # Example usage
    source_map = "map_two_rooms_18_19"
    target_map = "map_two_rooms_horizontally_18_40"
    
    vector_field = create_vector_field(source_map, target_map, target=False)
    #rotation_vectors(vector_field, vector_field.grid_length)

    target_vector_field = create_vector_field(source_map, target_map, target=True)

    compare_to_target(vector_field, target_vector_field, vector_field.grid_length)
    #compare_vector_direction(vector_field, vertical=False, range=45)



if __name__ == "__main__":
    main()