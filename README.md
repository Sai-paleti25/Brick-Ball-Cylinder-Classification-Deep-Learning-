# Deep Learning Programming Challenge


![](graphics/bbc1k.gif)

## About This Challenge

In the summer of 1966, Marvin Minsky and Seymour Paper, giants of Artifical Intelligence, launched the 1966 MIT Summer Vision Project: 

![](graphics/summer_project_abstract-01.png)

Minsky and Papert assigned Gerald Sussman, an MIT undergraduate studunt as project lead, and setup specific goals for the group around recognizing specific objects in images, and seperating these objects from their backgrounds. 

![](graphics/summer_project_goals-01.png)

Just how hard is it to acheive the goals Minsky and Papert laid out? How has the field of computer vision advance dsince that summer? Are these tasks trivial now, 50+ years later? Do we understand how the human visual system works? Just how hard *is* computer vision and how far have we come?

In this challenge, you'll use a modern tool, **deep neural networks**, and a labeled dataset to solve a version of the MIT Summer Vision Project problem.  

## Data
You'll be using the bbc-1k dataset, which contains 1000 images of bricks, balls, and cylinders against cluttered backgrounds. You can download the dataset [here](http://www.welchlabs.io/unccv/deep_learning/bbc_train.zip), or with the download script in the util directory of this repo:

```
python util/get_and_unpack.py -url http://www.welchlabs.io/unccv/deep_learning/bbc_train.zip
```

![](graphics/bbc_sample.jpg)

The BBC-1k dataset includes ~1000 images including classification, bounding box, and segmentation labels. Importantly, each image only contains one brick, ball or cylinder. 


## Packages
You are permitted to use numpy, opencv, tdqm, time, pytorch, fastai, opencv, and scipy.

## Your Mission 

Your job is to design and train a multitask deep learning model in fastai & pytorch to solve 3 different problems simultaneously: classification, detection, and segmentation. Please see `sample_student.py` and `eval.py` for more detailed instructions. A few tips: 

1. You will need to create a fastai dataloader the gives you all the labels you need (classifiction, bounding boxes, and segmentation). `eval.py` provides an example of one way to do this, by using fastai to load the segmentation mask, and then loading bounding boxes and classification labels from the mask. 
2. See the notebook `Get Results Fast with fastai.ipynb` for some examples to get you going, including a basic fastai multiclass network. 
3. It's worth spending some time thinkg through how you should "split apart" your network to solve multiple tastks. 


## Deliverables (Waiting for 2019 Updates)

1. Your modified version of `sample_student.py`. 
2. fastai export of our model file as a `.pkl` file, exported with `learn.export()`. 





