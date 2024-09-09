import matplotlib.pyplot as plt


def plot_loss(losses, title):
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    # Read the losses from the file

    with open("../agent/losses.txt", "r") as f:
        losses = f.readlines()

    losses = [float(loss.strip()) for loss in losses]
    print(losses)
    plot_loss(losses, "Losses")