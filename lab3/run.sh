#!/bin/bash

# for imageId in 06
# do
#     # echo "Running: img_00${imageId}_mask_Unet_toe.png"
#     python lab3.py "img_00${imageId}_mask_Unet_toe.png"
# done

for imageId in 06 07 09 10 12 15 16 20 21 27
do
    # echo "Running: img_00${imageId}_mask_Unet_toe.png"
    python lab3.py "img_00${imageId}_mask_Unet_toe.png"
done