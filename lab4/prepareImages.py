import os
import cv2
import numpy as np
import random

in_dir = r'./hiraganaPrepare'
out_dir = r'./data/hiragana/train/label'
out_dir_renamed = r'./data/hiragana/train/image'
test_dir = r'./data/hiragana/test'

def fillBackground(input_img):
    input_img = cv2.imread(input_img)

    for i in range(0, len(input_img)):
        for j in range(0, len(input_img[i])):
            if input_img[i][j][0] == 0 and input_img[i][j][1] == 0 and input_img[i][j][2] == 0:
                input_img[i][j][0] = random.randint(0, 255)
                input_img[i][j][1] = random.randint(0, 255)
                input_img[i][j][2] = random.randint(0, 255)

    # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    
    return input_img

def apply_mask(input_img):
    input_img = cv2.imread(input_img)

    for i in range(0, len(input_img)):
        for j in range(0, len(input_img[i])):
            if input_img[i][j][0] == 0 and input_img[i][j][1] == 0 and input_img[i][j][2] == 0:
                input_img[i][j][0] = 255
                input_img[i][j][1] = 255
                input_img[i][j][2] = 255
            else:
                input_img[i][j][0] = 0
                input_img[i][j][1] = 0
                input_img[i][j][2] = 0

    return input_img

if __name__ == "__main__":
    i = 0
    for filename in os.listdir(in_dir):
        out_name = str(i) + ".png"
        if i < 100:
            in_file = os.path.join(in_dir, filename)
            out_file = os.path.join(test_dir, out_name)
            input_img = cv2.imread(in_file)
            image_filled = fillBackground(in_file)
            cv2.imwrite(out_file, image_filled)
        else:
            in_file = os.path.join(in_dir, filename)
            out_file = os.path.join(out_dir, out_name)
            out_file_renamed = os.path.join(out_dir_renamed, out_name)

            image = apply_mask(in_file)
            cv2.imwrite(out_file, image)

            image_filled = fillBackground(in_file)

            cv2.imwrite(out_file_renamed, image_filled)
        i += 1
    