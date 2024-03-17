import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
from scipy.ndimage import binary_dilation


class FireSimulation:
    """A class to simulate the spread of fire using a grayscale image as input."""

    def __init__(self, image_path, radius=5, num_steps=500, fire_spread_prob=0.2):
        """
        Initialize the FireSimulation object.

        Parameters:
            image_path (str): The path to the grayscale image representing fire resistance.
            radius (int): The radius of the neighborhood for fire spread.
            num_steps (int): The number of simulation steps to run.
            fire_spread_prob (float): The probability of fire spreading to neighboring cells.
        """
        self.image_path = image_path
        self.radius = radius
        self.num_steps = num_steps
        self.fire_spread_prob = fire_spread_prob
        self.alpha_map = None
        self.kernel = None
        self.fire_matrix = None

    def load_image(self):
        """Load the grayscale image and convert it to an alpha coefficient map."""
        image = Image.open(self.image_path).convert("L")
        self.alpha_map = 1 - np.array(image) / 255.0

    def create_neumann_kernel(self):
        """Create a Neumann kernel for binary dilation based on the given radius."""
        size = 2 * self.radius + 1
        center = self.radius
        self.kernel = np.array(
            [
                [
                    1 if abs(i - center) + abs(j - center) <= self.radius else 0
                    for j in range(size)
                ]
                for i in range(size)
            ],
            dtype=int,
        )

    def update(self, step):
        """
        Update the fire matrix for the given time step.

        Parameters:
            step (int): The current time step of the simulation.
        """
        prev_fire = self.fire_matrix[:, :, step - 1].copy()
        dilated_matrix = binary_dilation(prev_fire, structure=self.kernel).astype(
            prev_fire.dtype
        )
        neighbor_matrix = dilated_matrix - prev_fire
        random_probabilities = np.random.rand(*prev_fire.shape)
        flammable_cells = neighbor_matrix * self.alpha_map * self.fire_spread_prob
        self.fire_matrix[:, :, step] = prev_fire + np.where(
            random_probabilities <= flammable_cells, 1, 0
        )

    def run_simulation(self):
        """Run the fire simulation."""
        self.load_image()
        self.create_neumann_kernel()

        self.fire_matrix = np.zeros(
            (self.alpha_map.shape[0], self.alpha_map.shape[1], self.num_steps)
        )
        initial_fire_pos = [800, 800]
        self.fire_matrix[initial_fire_pos[0], initial_fire_pos[1], 0] = 1

        for step in range(1, self.num_steps):
            self.update(step)

    def animate_simulation(self, save_path="fire_animation.gif"):
        """
        Animate the fire simulation and save it as a GIF.

        Parameters:
            save_path (str): The file path to save the animation as a GIF.
        """
        fig, ax = plt.subplots()
        plot = ax.imshow(self.fire_matrix[:, :, 0], cmap="Reds", alpha=0.5)

        def update_plot(frame):
            ax.set_title(f"Time Step: {frame}")
            plot.set_array(self.fire_matrix[:, :, frame])
            return (plot,)

        animation = FuncAnimation(
            fig, update_plot, frames=self.num_steps, interval=1, blit=True
        )
        animation.save(save_path, writer="pillow")
        plt.show()
        print("Animation saved")


if __name__ == "__main__":
    simulation = FireSimulation(
        image_path="/home/amir/Documents/NEFES/Simulation/map_fire_resistant_water.jpeg"
    )
    simulation.run_simulation()
    simulation.animate_simulation()
