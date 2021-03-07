#!/bin/bash
# Download training dataset
if [ ! -d "train2017" ] ; then
  echo -e "Downloading 2017 Train images [118K/18GB]"
  wget -c http://images.cocodataset.org/zips/train2017.zip
    
  echo -e "\nUnzipping dataset"
  unzip train2017.zip -d ./
    
  echo -e "\nRemove zip file"
  rm -rfv train2017.zip
else
  echo -e "train2017" dataset exists
fi

# Download validation dataset
if [ ! -d "val2017" ] ; then
  echo -e "2017 Val images [5K/1GB]"
  wget -c http://images.cocodataset.org/zips/val2017.zip
  
  echo -e "\nUnzipping dataset"
  unzip val2017.zip -d ./
  
  echo -e "\nRemove zip file"
  rm -rfv val2017.zip
else
  echo -e "val2017" dataset exists
fi

# Download testing dataset
if [ ! -d "test2017" ] ; then
  echo -e "2017 Test images [41K/6GB]"
  wget -c http://images.cocodataset.org/zips/test2017.zip
  
  echo -e "\nUnzipping dataset"
  unzip test2017.zip -d ./
  
  echo -e "\nRemove zip file"
  rm -rfv test2017.zip
else
  echo -e "test2017" dataset exists
fi

# Download train/val dataset annotations
if [ ! -d "annotations" ] ; then
  echo -e "2017 Train/Val annotations [241MB]"
  wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  
  echo -e "\nUnzipping dataset"
  unzip annotations_trainval2017.zip -d ./
  
  echo -e "\nRemove zip file"
  rm -rfv annotations_trainval2017.zip
else
  echo -e "annotations" dataset exists
fi
