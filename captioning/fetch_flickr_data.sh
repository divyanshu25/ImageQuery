#!/bin/bash

mkdir -p data
cd data
mkdir flickr
cd flickr

wget 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip'
unzip Flickr8k_Dataset.zip
rm -rf __MACOSX
rm Flickr8k_Dataset.zip

wget 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip'
unzip Flickr8k_text.zip
rm Flickr8k_text.zip
rm -rf __MACOSX
mkdir Flickr8k_text
mv *.txt Flickr8k_text


