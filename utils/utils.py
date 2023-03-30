import cv2


def img_orientation(im_size, size):
    if im_size[0] >= im_size[1]:
        return size
    else:
        return size[1], size[0]


def rescale_images(image_path):
    min_side = 1500
    max_side = 2000
    size = (min_side, max_side)
    img = cv2.imread(image_path)
    size_adj = img_orientation(img.shape, size)
    img_resized = cv2.resize(img, size_adj, interpolation=cv2.INTER_AREA)

    cv2.imwrite(image_path, img_resized)
    return img_resized
