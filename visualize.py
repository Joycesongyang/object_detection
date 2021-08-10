import cv2
from mscoco import get_data
import pdb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize(image, bboxes=None, masks=None, labels=None):
   pass


if __name__ == "__main__":
    image, bboxes, masks, labels = get_data()
    mask = masks[0]
    
    mask = np.stack([mask, mask, mask], axis=-1)
    mask = np.where(mask != [0, 0, 0], [75, 0, 130], image)
    mask = mask.astype(np.uint8)
    img_with_mask = cv2.addWeighted(image, 0.5, mask, 0.5, 0)

    # Create figure and axes
    fig,ax = plt.subplots(1,2, figsize=(15, 5))

    # Display the image
    ax[0].imshow(image)
    ax[1].imshow(img_with_mask)

    # Create a rectangle  patch
    rect = patches.Rectangle((bboxes[0][0],bboxes[0][1]),bboxes[0][2],bboxes[0][3],linewidth=1,edgecolor='g',facecolor='none')

    # Add the patch to the Axes
    ax[1].add_patch(rect)
    print("this is a {}".format(labels[0]))

    plt.show()
    