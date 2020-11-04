#!/bin/bash

wget 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip'
wget 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip'

mkdir -p data
cd data
unzip ../Flickr8k_Dataset.zip
mv Flicker8k_Dataset Flickr8k_Dataset
rm -rf __MACOSX
unzip ../Flickr8k_text.zip
rm -rf __MACOSX
mkdir Flickr8k_text
mv *.txt Flickr8k_text
rm ../Flickr8k_Dataset.zip
rm ../Flickr8k_text.zip
