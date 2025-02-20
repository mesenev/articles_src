import matplotlib.pyplot as plt
from PIL import Image, ImageSequence


images = []

for i in range(800):
    images.append(Image.open(f'1/{i}.png'))
images[0].save('animated_plot_1.gif', save_all=True, append_images=images)

images = []
for i in range(800):
    images.append(Image.open(f'2/{i}.png'))
images[0].save('animated_plot_2.gif', save_all=True, append_images=images)


# images = list()
# for i in range(1200):
#     im1 = Image.open(f'1/{i}.png')
#     im2 = Image.open(f'2/{i}.png')
#     dst = Image.new('RGB', (im1.width + im2.width, im1.height))
#     dst.paste(im1, (0, 0))
#     dst.paste(im2, (im1.width, 0))
#     images.append(dst)
#
# im1 = Image.open(f'animated_plot_1.gif')
# im2 = Image.open(f'animated_plot_2.gif')
# images = list()
# iter2 = ImageSequence.Iterator(im2)
# for frame1, frame2 in zip(ImageSequence.Iterator(im1), ImageSequence.Iterator(im2)):
#     dst = Image.new('RGB', (frame1.width + frame2.width, frame1.height))
#     dst.paste(frame1, (0, 0))
#     dst.paste(frame2, (im1.width, 0))
#     images.append(dst)
# images[0].save('animated.gif', save_all=True, append_images=images, duration=200, loop=0)
#
