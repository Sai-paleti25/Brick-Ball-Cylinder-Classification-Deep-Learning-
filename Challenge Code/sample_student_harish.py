## ---------------------------- ##
##
## sample_student.py
## Example student submission code for deep learning programming challenge. 
## You are free to use fastai, pytorch, opencv, and numpy for this challenge.
##
## Requirements:
## 0. Make sure your fastai model file is named export.pkl
## 1. Your code must be able to run on CPU or GPU.
## 2. Must handle different image sizes. 
## 3. Use a single unified pytorch model for all 3 tasks. 
## 
## ---------------------------- ##

from fastai.vision import load_learner, normalize, torch
import numpy as np


class Model(object):
    def __init__(self, path='../sample_models', file='export.pkl'):
        
        self.learn=load_learner(path=path, file=file) #Load model
        self.class_names=['background','brick', 'ball', 'cylinder'] #Be careful here, labeled data uses this order, but fastai will use alphabetical by default!

    def predict(self, x):
        '''
        Input: x = block of input images, stored as Torch.Tensor of dimension (batch_sizex3xHxW), 
                   scaled between 0 and 1. 
        Returns: a tuple containing: 
            1. The final class predictions for each image (brick, ball, or cylinder) as a list of strings.
            2. Upper left and lower right bounding box coordinates (in pixels) for the brick ball 
            or cylinder in each image, as a 2d numpy array of dimension batch_size x 4.
            3. Segmentation mask for the image, as a 3d numpy array of dimension (batch_sizexHxW). Each value 
            in each segmentation mask should be either 0, 1, 2, or 3. Where 0=background, 1=brick, 
            2=ball, 3=cylinder. 
        '''

        #Normalize input data using the same mean and std used in training:
        x_norm=normalize(x, torch.tensor(self.learn.data.stats[0]), 
                            torch.tensor(self.learn.data.stats[1]))


        #Pass data into model:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            yhat=self.learn.model(x_norm.to(device))
            yhat=yhat.detach().cpu()

        #Post-processing/parsing outputs, here's an example for classification only:
        class_prediction_indices=yhat.argmax(1)
        
        
        #retrieving class values from yhat
        class_values = np.zeros(class_prediction_indices.shape[0], dtype = 'int8')
        for i in range(class_prediction_indices.shape[0]):
          cls,values = np.unique(class_prediction_indices[i], return_counts=True)
          counterlist = list(zip(cls,values))
          count = sorted(counterlist,key = lambda item:item[1], reverse=True)
          if len(count)==1:
            class_values[i] = count[0][0]
          else:
            class_values[i] = count[1][0]

        #creating class label strings
        #print(class_values)
        class_label_strings = [self.class_names[i] for i in class_values]
        print(class_label_strings)

        
        #Create random segmentation mask:
        # mask=np.random.randint(low=0, high=4, size=(x.shape[0], x.shape[2], x.shape[3]))
        bboxes=np.zeros((len(class_prediction_indices), 4))
        for i in range(len(class_prediction_indices)):
            rows,cols= np.where(class_prediction_indices[i]!=0)
            bboxes[i, :] = np.array([rows.min(), cols.min(), rows.max(), cols.max()])

        class_prediction_indices = np.array(class_prediction_indices)
        return (class_label_strings, bboxes, class_prediction_indices)
