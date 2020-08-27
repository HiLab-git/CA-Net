import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def map_scalar_to_color(x):
    x_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    c_list = [[0, 0, 255],
              [0, 255, 255],
              [0, 255, 0],
              [255, 255, 0],
              [255, 0, 0]]
    for i in range(len(x_list)):
        if(x <= x_list[i + 1]):
            x0 = x_list[i]
            x1 = x_list[i + 1]
            c0 = c_list[i]
            c1 = c_list[i + 1]
            alpha = (x - x0)/(x1 - x0)
            c = [c0[j]*(1 - alpha) + c1[j] * alpha for j in range(3)]
            c = [int(item) for item in c]
            return tuple(c)


def get_fused_heat_map(image, att):
    [H, W] = image.size
    img = Image.new('RGB', image.size, (255, 0, 0))
    
    for i in range(H):
        for j in range(W):
            p0 = image.getpixel((i,j))
            alpha = att.getpixel((i,j))
            p1 = map_scalar_to_color(alpha)
            alpha = 0.3 + alpha*0.5
            p = [int(p0[c] * (1 - alpha) + p1[c]*alpha) for c in range(3)]
            p = tuple(p)
            img.putpixel((i, j), p)
    return img


if __name__ == "__main__":
    image_name = "./result/atten_map/ISIC_0015937.jpg"
    scalar_name = "./result/atten_map/25_2_8_wgt"
    save_name = "./result/atten_map/15937_wgt3_fused"

    img = Image.open(image_name)
    # img = np.load(image_name)
    # img = Image.fromarray(np.uint8(img*255))
    # load the scalar map, and normalize the inteinsty to  0 - 1
    scl = Image.open(scalar_name).convert('L')
    scl = np.asarray(scl)
    scl = cv2.resize(scl, dsize=(img.size[0], img.size[1]), interpolation=cv2.INTER_NEAREST)
    scl_norm = np.asarray(scl, np.float32)/255
    scl_norm = Image.fromarray(scl_norm)
    
    # convert the scalar map to heat map, and fuse it with the original image
    img_scl = get_fused_heat_map(img, scl_norm)
    # img_scl.save(save_name, format='png')

    plt.imshow(img_scl), plt.title('fused result')
    # plt.colorbar()
    plt.show()
