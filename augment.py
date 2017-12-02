import numpy as np
import cv2


def flip(img, label):
    return cv2.flip(img, 1), cv2.flip(label, 1)


def random_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    random_bright = np.random.uniform(0.3, 1.7)
    hsv[:, :, 2] = hsv[:, :, 2] * random_bright
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def random_crop(img, label):
    img_h, img_w, _ = img.shape
    aspect_ratio = img_w / img_h
    top = np.random.randint(0, 100)
    bottom = np.random.randint(0, 50)
    left = np.random.randint(0, 350)
    h = img_h - top - bottom
    w = int(aspect_ratio * h)
    if w > img_w - left:
        w = img_w - left
        h = int(w / aspect_ratio)
    return img[top:top+h, left:left+w, :], label[top:top+h, left:left+w, :]


def augment_image(img, label):
    new_img, new_label = img, label
    if np.random.rand() > 0.5:
        new_img, new_label = flip(new_img, new_label)

    new_img, new_label = random_crop(new_img, new_label)
    new_img = random_brightness(new_img)
    return new_img, new_label

