import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from typing import Tuple


def apply_depth_colormap(gray, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    Input:
        gray: gray image, tensor/numpy, (H, W)
    Output:
        depth: (3, H, W), tensor
    """
    if type(gray) is not np.ndarray:
        gray = gray.detach().cpu().numpy().astype(np.float32)
    gray = gray.squeeze()
    assert len(gray.shape) == 2
    x = np.nan_to_num(gray)  # change nan to 0
    if minmax is None:
        mi = np.min(x)  # get minimum positive value
        ma = np.max(x)
    else:
        mi, ma = minmax
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    # TODO
    x = 1 - x  # reverse the colormap
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(cv2.applyColorMap(x, cmap))
    x = T.ToTensor()(x)  # (3, H, W)
    return x


def apply_semantic_colormap(semantic):
    """
    Input:
        semantic: semantic image, tensor/numpy, (N, H, W)
    Output:
        depth: (3, H, W), tensor
    """

    color_id = np.zeros((20, 3), dtype=np.uint8)
    # color_id[0, :] = [255, 255, 255]  # 0 undefined
    # color_id[1, :] = [112, 128, 144]  # 1 barrier  orange
    # color_id[2, :] = [220, 20, 60]    # 2 bicycle  Blue
    # color_id[3, :] = [255, 127, 80]   # 3 bus  Darkslategrey
    # color_id[4, :] = [255, 158, 0]    # 4 car  Crimson
    # color_id[5, :] = [233, 150, 70]   # 5 cons. Veh  Orangered
    # color_id[6, :] = [255, 61, 99]    # 6 motorcycle  Darkorange
    # color_id[7, :] = [0, 0, 230]      # 7 pedestrian  Darksalmon
    # color_id[8, :] = [47, 79, 79]     # 8 traffic cone  Red
    # color_id[9, :] = [255, 140, 0]    # 9 trailer  Slategrey
    # color_id[10, :] = [255, 99, 71]   # 10 truck Burlywood
    # color_id[11, :] = [0, 207, 191]   # 11 drive sur  Green
    # color_id[12, :] = [175, 0, 75]    # 12 other lat  nuTonomy green
    # color_id[13, :] = [75, 0, 75]     # 13 sidewalk
    # color_id[14, :] = [112, 180, 60]  # 14 terrain
    # color_id[15, :] = [222, 184, 135] # 15 manmade
    # color_id[16, :] = [0, 175, 0]     # 16 vegetation
    # color_id[17, :] = [0, 255, 255]  # 17 ped_crossing
    # color_id[18, :] = [255, 0, 255]    # 18 road_divider
    # color_id[19, :] = [0, 255, 0]   # 19 lane_divider

    color_id[0, :] = [255, 255, 255]
    color_id[1, :] = [255, 192, 203]
    color_id[2, :] = [255, 255, 0]
    color_id[3, :] = [0, 150, 245]
    color_id[4, :] = [255, 158, 0]
    color_id[5, :] = [255, 127, 0]
    color_id[6, :] = [255, 0, 0]
    color_id[7, :] = [0, 0, 255]
    color_id[8, :] = [135, 60, 0]
    color_id[9, :] = [160, 32, 240]
    color_id[10, :] = [255, 0, 255]
    color_id[11, :] = [139, 137, 137]
    color_id[12, :] = [75, 0, 75]
    color_id[13, :] = [255, 192, 0]
    color_id[14, :] = [230, 230, 250]
    color_id[15, :] = [222, 184, 135]
    color_id[16, :] = [0, 175, 0]
    color_id[17, :] = [75, 0, 75]
    color_id[18, :] = [233, 113, 50]
    color_id[19, :] = [233, 113, 50]

    if semantic.shape[0] != 1:
        semantic = torch.max(semantic, dim=0)[1].squeeze()
    else:
        semantic = semantic.squeeze()

    x = torch.zeros((3, semantic.shape[0], semantic.shape[1]), dtype=torch.float)
    # first stuff
    for i in range(11, 20):
        x[0][semantic == i] = color_id[i][0]
        x[1][semantic == i] = color_id[i][1]
        x[2][semantic == i] = color_id[i][2]
    # then instance
    for i in range(0, 11):
        x[0][semantic == i] = color_id[i][0]
        x[1][semantic == i] = color_id[i][1]
        x[2][semantic == i] = color_id[i][2]

    return x / 255.0


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

def vis_occ_render(images, occ_render_images, occ_render_depth):
    # images = 6,3,224,400
    H, W = images.shape[-2:]
    camera_images = []
    for img in images:
        camera_image = img_m11_to_01(img).cpu().permute(1, 2, 0).detach().numpy() * 255
        camera_image = Image.fromarray(camera_image.astype(np.uint8))
        camera_images.append(camera_image)

    # occ_render_images = 6,20,448,800
    semantic_images = []
    for img in occ_render_images:
        semantic_image = apply_semantic_colormap(img.argmax(0).unsqueeze(0))
        semantic_image = semantic_image.cpu().permute(1, 2, 0).detach().numpy() * 255
        semantic_image = Image.fromarray(semantic_image.astype(np.uint8))
        semantic_image = semantic_image.resize((W, H), resample=Image.NEAREST)
        semantic_images.append(semantic_image)

    # occ_render_depth = 6,1,448,800
    depth_images = []
    for img in occ_render_depth:
        depth_image = apply_depth_colormap(img).cpu().permute(1, 2, 0).detach().numpy() * 255
        depth_image = Image.fromarray(depth_image.astype(np.uint8))
        depth_image = depth_image.resize((W, H), resample=Image.NEAREST)
        depth_images.append(depth_image)
    

    # 透明度
    alpha = 0.3

    semantic_images = concat_6_views(semantic_images)
    semantic_images = semantic_images.convert("RGBA")
    semantic_with_alpha = semantic_images.copy()
    alpha_channel = semantic_with_alpha.split()[3].point(lambda p: int(p * alpha))
    semantic_with_alpha.putalpha(alpha_channel)

    depth_images = concat_6_views(depth_images)
    depth_images = depth_images.convert("RGBA")
    depth_with_alpha = depth_images.copy()
    alpha_channel = depth_with_alpha.split()[3].point(lambda p: int(p * alpha))
    depth_with_alpha.putalpha(alpha_channel)

    camera_images = concat_6_views(camera_images)
    camera_images = camera_images.convert("RGBA")
    blended_semantic = Image.alpha_composite(camera_images, semantic_with_alpha)
    blended_depth = Image.alpha_composite(camera_images, depth_with_alpha)
    blended_semantic.save("./semantic_color.png")
    blended_depth.save("./depth_color.png")