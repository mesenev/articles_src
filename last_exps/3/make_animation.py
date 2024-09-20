import matplotlib.pyplot as plt
from PIL import Image

images = []
for i in range(1000):
    images.append(Image.open(f'animation/{i}.png'))
images[0].save('animated_plot.gif', save_all=True, append_images=images, duration=200, loop=0)