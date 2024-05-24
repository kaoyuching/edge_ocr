import numpy as np
import cv2


# load data util functions
def load_detect_image(img_path: str):
    r'''
    load input image
    '''
    img = cv2.imread(img_path)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    img_orig_shape = img.shape[:2] # h,w
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)  # resize (args?)
    img = img/255.
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)  # (n, c, h, w)
    img_shape = img.shape[2:]
    return img, img_orig_shape, img_shape


# resize bounding box to fit original image size
def decode_box(coords: np.ndarray, img_size: tuple, img_new_size: tuple, padding: tuple = (0, 0)):
    # img_size: (h, w)
    # img_new_size: (h, w)
    # padding: (h,w)
    h, w  = img_new_size
    scale_h = h/(img_size[0] - padding[0]*2)
    scale_w = w/(img_size[1] - padding[1]*2)

    new_coords = np.copy(coords)

    center_x = (new_coords[:, 0] + new_coords[:, 2]) / 2 - padding[1]
    center_y = (new_coords[:, 1] + new_coords[:, 3]) / 2 - padding[0]
    half_w = (new_coords[:, 2] - new_coords[:, 0]) / 2
    half_h = (new_coords[:, 3] - new_coords[:, 1]) / 2

    new_coords[:, 0] = (center_x*scale_w - half_w*scale_w).clip(0, w)
    new_coords[:, 1] = (center_y*scale_h - half_h*scale_h).clip(0, h)
    new_coords[:, 2] = (center_x*scale_w + half_w*scale_w).clip(0, w)
    new_coords[:, 3] = (center_y*scale_h + half_h*scale_h).clip(0, h)
    return new_coords


def extend_box(bbox: tuple, ratio: float, w: int, h: int):
    xmin, ymin, xmax, ymax = bbox
    center_x = (xmin + xmax)/2
    center_y = (ymin + ymax)/2
    hw = (xmax - xmin)/2
    hh = (ymax - ymin)/2
    e_xmin = max(center_x - hw*ratio, 0)
    e_xmax = min(center_x + hw*ratio, w)
    e_ymin = max(center_y - hh*ratio, 0)
    e_ymax = min(center_y + hh*ratio, h)
    return e_xmin, e_ymin, e_xmax, e_ymax


def load_ocr_image(img_path: str, bbox: tuple, extend_ratio: float = 1.15) -> np.ndarray:
    img = cv2.imread(img_path)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # BGR -> GRAY
    h, w = img.shape
    # extend box and crop image
    xmin, ymin, xmax, ymax = extend_box(bbox, ratio=extend_ratio, w=w, h=h)
    img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
    cv2.imwrite('_test123_bbox.png', img)
    img = np.array(img)
    # stack and normalilze
    img = np.stack((img,)*3, axis=-1).astype(np.uint8)/255
    # resize
    img = cv2.resize(img, (512, 64), interpolation=cv2.INTER_LINEAR)  # resize (w, h)
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)  # (n, c, h, w)
    return img.astype(np.float32)
