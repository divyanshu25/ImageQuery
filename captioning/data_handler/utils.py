#  ================================================================
#  Copyright 2020 Image Query Team
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==================================================================
from pycocotools.coco import COCO


def parse_flickr(ann_file):
    ann_dict = {}
    with open(ann_file, "r") as f:
        lines = f.read().splitlines()
        for l in lines:
            tokens = l.split("\t", 1)
            # print(tokens)
            ann_dict[tokens[0]] = tokens[1]
    return ann_dict


def parse_coco(ann_file):
    coco = COCO(ann_file)
    ann_dict = {}
    for id in coco.imgs.keys():
        ann_ids = coco.getAnnIds(imgIds=id)
        anns = coco.loadAnns(ann_ids)
        target = [ann["caption"] for ann in anns]
        ann_dict[id] = target
    return ann_dict
