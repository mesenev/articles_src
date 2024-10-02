import matplotlib.pyplot as plt
from PIL import Image

images = []
for i in range(1200):
    images.append(Image.open(f'1/{i}.png'))
images[0].save('animated_plot_1.gif', save_all=True, append_images=images, duration=200, loop=0)

images = []
for i in range(1200):
    images.append(Image.open(f'2/{i}.png'))
images[0].save('animated_plot_2.gif', save_all=True, append_images=images, duration=200, loop=0)
