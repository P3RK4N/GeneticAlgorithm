# import cv2
import numpy as np
import torch

MIN_I = 0
MAX_I = 1000
MIN_J = 0
MAX_J = 1000
DELAY = 1
resolution = (MAX_I-MIN_I, MAX_J-MIN_J)



def xy2ij(x):
    a, b = x[:, 0:1], x[:, 1:]
    return np.concatenate((MAX_I-b, a), axis=1)

def ij2xy(x):
    a, b = x[:, 0:1], x[:, 1:]
    return np.concatenate((b, MAX_I-a), axis=1)

def ij2xy_cv2(x):
    a, b = x[:, 0:1], x[:, 1:]
    return np.concatenate((b, a), axis=1)

def draw_body(x, rods, rod_colors):
    img = np.zeros(resolution, dtype=np.uint8)
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    x = x.astype(int)
    x = xy2ij(x)
    x -= np.array([[MIN_I, MIN_J]])
    x[:, 0] = np.clip(x[:, 0], 1, resolution[0]-2)
    x[:, 1] = np.clip(x[:, 1], 1, resolution[1]-2)


    for i in range(-1, 2):
        for j in range(-1, 2):
            y = x - np.array([[i, j]])
            img[y[:,0], y[:, 1]] = 255

    xy_pts = ij2xy_cv2(x)
    for c, (ix, iy) in zip(rod_colors, rods):
        img = cv2.line(img, tuple(xy_pts[ix]), tuple(xy_pts[iy]), color=int(c))

    return img


def show_body(x, rods, rod_colors, text=None):
    img = draw_body(x, rods, rod_colors)
    if text is not None:
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = 255
        thickness = 2
        img = cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('hello', img)
    if cv2.waitKey(DELAY) == ord('q'):
        raise Exception('you wanted to exit')
    return img