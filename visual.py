import cv2
from mscoco import get_data
import pdb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pycocotools.coco import COCO
import os
import json
import random
import colorsys

#generate mask color
def auto_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

 #generate box color   
def random_color():
    h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
    r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
    return '#%02X%02X%02X' % (r, g, b)

def top_left(num):
    row_sum = np.sum(num, axis = 1)
    colm_sum = np.sum(num, axis = 0)

    for ind, i in enumerate(row_sum):
        if i > 1:
            row_sum[ind] = 1 
    for ind, i in enumerate(colm_sum):
        if i >1:
            colm_sum[ind] = 1

    top_x, top_y = colm_sum.tolist(),row_sum.tolist()
    
    return top_x.index(1), top_y.index(1)



def visualize(image, bboxes=None, masks=None, labels=None):
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # Create figure and axes
    fig,ax = plt.subplots(1,2, figsize=(15, 5))
    # Display the image
    ax[0].imshow(image)
    img_with_mask = image

    if bboxes is not None:
        a = len(bboxes)
        for i in range (a):
            if masks is not None: 
                mask = masks[i]
                mask = np.stack([mask, mask, mask], axis=-1)
                # mask = np.where(mask != [0, 0, 0], [75, 0, 130], img_with_mask)
                mask = np.where(mask != [0, 0, 0], auto_color(), img_with_mask)
                mask = mask.astype(np.uint8)
                img_with_mask = cv2.addWeighted(img_with_mask, 0.5, mask, 0.5, 0)

            ax[1].imshow(img_with_mask)

            # Create a rectangle  patch
            rect = patches.Rectangle((bboxes[i][0],bboxes[i][1]),bboxes[i][2],bboxes[i][3],linewidth=1,edgecolor=random_color(),facecolor='none')

            # Add the patch to the Axes
            ax[1].add_patch(rect)
    
            # Add label to the bbox
            if labels is not None:
                ax[1].text(bboxes[i][0] + 3, bboxes[i][1] + 16, str(labels[i]), color='red', fontsize=10)
            
    else:
        if masks is not None:
            a  =len(masks)
            for i in range (a):
                mask = masks[i]
                mask = np.stack([mask, mask, mask], axis=-1)
                mask = np.where(mask != [0, 0, 0], auto_color(), img_with_mask)
                mask = mask.astype(np.uint8)
                img_with_mask = cv2.addWeighted(img_with_mask, 0.5, mask, 0.5, 0)

                ax[1].imshow(img_with_mask)

                if labels is not None:
                    ax[1].text(top_left(masks[i])[0], top_left(masks[i])[1], str(labels[i]), color='red', fontsize=10)
            
        else:
            ax[1].imshow(img_with_mask)
            plt.title((labels),fontdict={'fontsize' : 10}, loc='center', pad=None) 




    plt.show()


if __name__ == "__main__":
    # 1 1 1 1
    image, bboxes, masks, labels = get_data(read_image=True, bbox=True, mask=True, label=True)
    # 1 1 1 0
    # image, bboxes, masks, labels = get_data(read_image=True, bbox=True, mask=True, label=False)
    # 1 1 0 1
    # image, bboxes, masks, labels = get_data(read_image=True, bbox=True, mask=False, label=True)
    # # 1 1 0 0
    # image, bboxes, masks, labels = get_data(read_image=True, bbox=True, mask=False, label=False)
    # # 1 0 1 1 
    # image, bboxes, masks, labels = get_data(read_image=True, bbox=False, mask=True, label=True)
    # # 1 0 1 0
    # image, bboxes, masks, labels = get_data(read_image=True, bbox=False, mask=True, label=False)
    # # 1 0 0 1
    # image, bboxes, masks, labels = get_data(read_image=True, bbox=False, mask=False, label=True)
    # # 1 0 0 0
    # image, bboxes, masks, labels = get_data(read_image=True, bbox=False, mask=False, label=False)
    # # 0 1 1 1
    # image, bboxes, masks, labels = get_data(read_image=False, bbox=True, mask=True, label=True)
    # # 0 1 1 0
    # image, bboxes, masks, labels = get_data(read_image=False, bbox=True, mask=True, label=False)
    # # 0 1 0 1
    # image, bboxes, masks, labels = get_data(read_image=False, bbox=True, mask=False, label=True)
    # # 0 1 0 0
    # image, bboxes, masks, labels = get_data(read_image=False, bbox=True, mask=False, label=False)
    # 0 0 1 1
    # image, bboxes, masks, labels = get_data(read_image=False, bbox=False, mask=True, label=True)
    # 0 0 0 0
    # image, bboxes, masks, labels = get_data(read_image=False, bbox=False, mask=False, label=False)
    # 0 0 0 1
    # image, bboxes, masks, labels = get_data(read_image=False, bbox=False, mask=False, label=True)
    # 0 0 1 0
    # image, bboxes, masks, labels = get_data(read_image=False, bbox=False, mask=True, label=False)


    visualize(image, bboxes, masks, labels)



