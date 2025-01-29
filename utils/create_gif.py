from PIL import Image

def create_gif(gif_path: str, frames: list):
        """
        Creates a GIF from a list of frames.

        Args:
            frames (list): A list of frames to be included in the GIF.
            gif_path (str): The path to save the GIF file.
            duration (int): The duration of each frame in milliseconds.

        Returns:
            None
        """
        images = [Image.fromarray(frame, mode="RGB") for frame in frames]
        images[0].save(
            gif_path, save_all=True, append_images=images[1:], duration=100, loop=0
        )
        return gif_path