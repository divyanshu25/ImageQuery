import os
import sys

sys.path.append("/opt/cocoapi/PythonAPI")
from pycocotools.coco import COCO

# initialize COCO API for instance annotations
dataDir = "/opt/cocoapi"
dataType = "val2014"
instances_annFile = os.path.join(
    dataDir, "annotations/instances_{}.json".format(dataType)
)
coco = COCO(instances_annFile)

# initialize COCO API for caption annotations
captions_annFile = os.path.join(
    dataDir, "annotations/captions_{}.json".format(dataType)
)
coco_caps = COCO(captions_annFile)

# get image ids
ids = list(coco.anns.keys())
