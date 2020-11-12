#!/bin/bash

mkdir -p data
cd data
mkdir -p coco
cd coco
# Get Train
curl http://images.cocodataset.org/zips/train2017.zip>train2017.zip
unzip train2017.zip
rm train2017.zip

# Get Val
curl http://images.cocodataset.org/zips/val2017.zip>val2017.zip
unzip val2017.zip
rm val2017.zip

# Get Test
curl http://images.cocodataset.org/zips/test2017.zip>test2017.zip
unzip test2017.zip
rm test2017.zip

curl http://images.cocodataset.org/annotations/annotations_trainval2017.zip>annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip
mv annotations/* .

curl http://images.cocodataset.org/annotations/image_info_test2017.zip > image_info_test2017.zip
unzip image_info_test2017.zip
rm image_info_test2017.zip
mv annotations/* .
rmdir annotations
