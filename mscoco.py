from pycocotools.coco import COCO
import numpy as np
import os
import cv2
import json

def get_data(read_image=True, bbox=True, mask=True, label=True):
    data_dir = "/Users/Joyce/Desktop/file/project/object_detection/data/val2017"
    annotation_path = "/Users/Joyce/Desktop/file/project/object_detection/data/annotations/instances_val2017.json"
    class_name_path = "/Users/Joyce/Desktop/file/project/object_detection/classes.json"
    with open(class_name_path, 'r') as f:
        class_map = json.load(f)
    coco=COCO(annotation_path)
    imgIds = coco.getImgIds()
    imgId = imgIds[int(np.random.randint(0,len(imgIds)))]
    annIds = coco.getAnnIds(imgIds=imgId)
    anns = coco.loadAnns(annIds)
    image = os.path.join(data_dir, coco.loadImgs(imgId)[0]["file_name"])
    bboxes = [] if bbox else None
    masks = [] if mask else None
    labels = [] if label else None
    if read_image:
        image = cv2.imread(image)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    if bbox or mask or label:
        for annotation in anns:
            if bbox:
                bboxes.append(annotation["bbox"])
            if mask:
                masks.append(coco.annToMask(annotation))
            if label:
                labels.append(class_map[str(annotation['category_id'])])
    return image, bboxes, masks, labels
