from typing import Tuple
from PIL import Image
import numpy as np


def concat_6_views(imgs: Tuple[Image.Image, ...], oneline=False):
    if oneline:
        image = img_concat_h(*imgs)
    else:
        image = img_concat_v(img_concat_h(*imgs[:3]), img_concat_h(*imgs[3:]))
    return image


def img_m11_to_01(img):
    return img * 0.5 + 0.5


def img_concat_h(im1, *args, color='black'):
    if len(args) == 1:
        im2 = args[0]
    else:
        im2 = img_concat_h(*args)
    height = max(im1.height, im2.height)
    mode = im1.mode
    dst = Image.new(mode, (im1.width + im2.width, height), color=color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def img_concat_v(im1, *args, color="black"):
    if len(args) == 1:
        im2 = args[0]
    else:
        im2 = img_concat_v(*args)
    width = max(im1.width, im2.width)
    mode = im1.mode
    dst = Image.new(mode, (width, im1.height + im2.height), color=color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def visualize_image_with_canvas(image_list, canvas_list):
    # image: 6,3,H,W
    # canvas: 6,C,H,W
    image_list = np.transpose(image_list, (0, 2, 3, 1))
    # (-1,1) - > (0,1)
    image_list = img_m11_to_01(image_list)

    images_ = []
    colors =[(1,0,0),(0,1,0),(0,0,1)]
    for image, canvas in zip(image_list, canvas_list):
        for category in range(3):
            image[canvas[category] == 1] = [colors[category][0], colors[category][1], colors[category][2]]
        image = (image * 255).astype(np.uint8) 
        images_.append(Image.fromarray(image))

    concat_images = concat_6_views(images_)
    concat_images.save("output_image.png")
    return concat_images