
import numpy as np

random_colours = np.random.rand(300, 12)
print(np.min(random_colours), np.mean(random_colours), np.max(random_colours))
np.save("../random_palettes.npy", random_colours)

