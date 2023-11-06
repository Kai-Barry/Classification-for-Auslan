from moduleExperiment8 import convert_one_video, reshape_x, get_example
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

location = 'experiment8/unprocessed/SYDNEY_MTFB3c9a_317220_319465.npy'
locationShape = 'experiment8/unprocessed/SYDNEY_MTFB3c9a_317220_319465_shape.npy'

example = get_example(location, locationShape)
xy_frames, xz_frames, yz_frames = convert_one_video(example, 128)

def update(frame):
    # Update the images with the next frame from each set
    im1.set_data(xy_frames[frame])
    im2.set_data(xz_frames[frame])
    im3.set_data(yz_frames[frame])


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))  # 1 row, 2 columns

ax1.axis('off')
ax2.axis('off')
ax3.axis('off')


num_frames = 40  # Number of frames (the number of images in your array)
ani_interval = 25  # Interval between frames in milliseconds

im1 = ax1.imshow(xy_frames[0], cmap='gray', origin='lower')
im2 = ax2.imshow(xz_frames[0], cmap='gray', origin='lower')
im3 = ax3.imshow(yz_frames[0], cmap='gray', origin='lower')

ani = FuncAnimation(fig, update, frames=num_frames, repeat=True, interval=100)

plt.show()
