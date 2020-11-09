import os
import sys
import cv2
import numpy as np
import random


def augment_image(input_img, mode):
    input_img = cv2.imread(input_img)

    if mode == 2:
        bright = np.ones(input_img.shape, dtype="uint8") * 70

        return cv2.subtract(input_img, bright)
    elif mode == 3:
        sharpening = np.array([[-1, -1, -1, [-1, 10, -1], [-1, -1, -1]]])

        return cv2.filter2D(image, -1, sharpening)
    else:
        bright = np.ones(input_img.shape, dtype="uint8") * 70

        return cv2.add(input_img, bright)


if __name__ == "__main__":
    in_dir = "./dataset/" + sys.argv[1]
    out_dir = "./dataset_new/" + sys.argv[1]
    augment_mode = sys.argv[2]

    for data_inner_dir in os.listdir(in_dir):
        try:
            for filename in os.listdir(os.path.join(in_dir, data_inner_dir)):
                if str(filename).endswith(".png"):
                    in_file = os.path.join(
                        os.path.join(in_dir, data_inner_dir), filename
                    )
                    out_file = os.path.join(
                        os.path.join(out_dir, data_inner_dir), filename
                    )

                    try:
                        os.remove(out_file)
                    except FileNotFoundError:
                        pass

                    image = augment_image(in_file, augment_mode)
                    cv2.imwrite(out_file, image)
        except NotADirectoryError:
            pass