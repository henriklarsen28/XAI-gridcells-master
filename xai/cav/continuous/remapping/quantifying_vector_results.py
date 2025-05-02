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



def compare_to_target(vector_field: VectorField, target_vector_field: VectorField, grid_length: int):
    pass

def main():
    # Example usage
    
    vector_field = create_vector_field()
    rotation_vectors(vector_field, vector_field.grid_length)

    target_vector_field = create_vector_field(target=True)



if __name__ == "__main__":
    main()