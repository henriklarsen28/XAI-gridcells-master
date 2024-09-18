import matplotlib.pyplot as plt
import pandas as pd


def plot_loss(losses, title):
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    # Read the losses from the file


    df = pd.read_csv("../agent/losses_rewards.csv")

    losses = df["loss"].values
    plot_loss(losses, "Losses")