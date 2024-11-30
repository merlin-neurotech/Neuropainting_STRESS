import numpy as np
import matplotlib.pyplot as plt
import random

# Function to generate random colors, adjusted for stress level
def random_color(stress_level):
    if stress_level < 0:
        # Darker and more intense colors for negative stress levels
        return np.clip(np.random.rand(3,) * (0.5 + (stress_level / -10)), 0, 1)
    else:
        # Brighter and more harmonious colors for positive stress levels
        return np.clip(np.random.rand(3,) * (0.5 + (stress_level / 10)) + 0.5, 0, 1)

# Function to generate random geometric shapes
def random_shapes(ax, num_shapes, stress_level):
    for _ in range(num_shapes):
        # More jagged, chaotic shapes for negative stress levels
        shape_type = random.choices(
            ['circle', 'rectangle', 'triangle', 'line', 'polygon'],
            weights=[1 + stress_level if stress_level > 0 else 1, 
                     1 + stress_level if stress_level > 0 else 1,
                     1 + abs(stress_level) if stress_level < 0 else 1,
                     abs(stress_level) if stress_level < 0 else 0.5,
                     abs(stress_level) if stress_level < 0 else 0.5])[0]
        
        color = random_color(stress_level)
        
        if shape_type == 'circle':
            radius = random.uniform(0.05, 0.3)
            x, y = random.uniform(0, 1), random.uniform(0, 1)
            circle = plt.Circle((x, y), radius, color=color, fill=True)
            ax.add_patch(circle)
        elif shape_type == 'rectangle':
            width, height = random.uniform(0.1, 0.4), random.uniform(0.1, 0.4)
            x, y = random.uniform(0, 1 - width), random.uniform(0, 1 - height)
            rectangle = plt.Rectangle((x, y), width, height, color=color, fill=True)
            ax.add_patch(rectangle)
        elif shape_type == 'triangle':
            points = np.random.rand(3, 2)
            triangle = plt.Polygon(points, color=color, fill=True)
            ax.add_patch(triangle)
        elif shape_type == 'line':
            # Chaotic lines for higher stress
            x1, y1 = random.uniform(0, 1), random.uniform(0, 1)
            x2, y2 = random.uniform(0, 1), random.uniform(0, 1)
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=3)
        elif shape_type == 'polygon':
            # Random multi-point polygon for stress
            num_sides = random.randint(3, 6)
            points = np.random.rand(num_sides, 2)
            polygon = plt.Polygon(points, color=color, fill=True)
            ax.add_patch(polygon)

# Function to generate random painting with stress level
def generate_painting(stress_level, num_shapes=15):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    # Adjust the randomness of shapes based on stress level
    random_shapes(ax, num_shapes, stress_level)

    # Adjust the background based on stress level (darker for stress)
    if stress_level < 0:
        bg_color = np.clip(1 + (stress_level / 10), 0, 1)
        fig.patch.set_facecolor((bg_color, bg_color, bg_color))
    else:
        bg_color = np.clip(1 - (stress_level / 10), 0, 1)
        fig.patch.set_facecolor((1, 1, 1))

    plt.show()

# Generate a random stress level between -10 and 10
stress_level = random.randint(-10, 10)
print(f"Generated stress level: {stress_level}")

# Generate a painting based on the stress level
generate_painting(stress_level, num_shapes=20)
