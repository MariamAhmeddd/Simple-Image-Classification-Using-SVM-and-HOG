#!/usr/bin/env python
# coding: utf-8

# In[9]:


from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn import svm
from sklearn.metrics import accuracy_score
import os


# In[10]:


train_f = r'E:\bahgt talat 5ales\Semester 6\Pattern\Labs\lab8\Assignment dataset\train'
test_f = r'E:\bahgt talat 5ales\Semester 6\Pattern\Labs\lab8\Assignment dataset\test'


# In[11]:


orientations = 9  # Number of gradient orientations
pixels_per_cell = (8, 8)  # Size of each cell
cells_per_block = (2, 2)  # Number of cells in each block


# In[27]:


def extract_features(image_path):
    image = imread(image_path, as_gray=True)
    resized_image = resize(image, (64, 64)) # Resize the image to a fixed size
    fd = hog(resized_image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=False)
    return fd


# In[28]:


train_features = []
train_target = []

for folder in os.listdir(train_f):
    class_f = os.path.join(train_f, folder)
    if os.path.isdir(class_f):
        for image_name in os.listdir(class_f):
            image_path = os.path.join(class_f, image_name)
            features = extract_features(image_path)
            train_features.append(features)
            
            train_target.append(folder)


# In[29]:


svm_model = svm.SVC(kernel='linear')
svm_model.fit(train_features, train_target)


# In[30]:


test_features = []
test_target = []

for folder in os.listdir(test_f):
    class_f = os.path.join(test_f, folder)
    if os.path.isdir(class_f):
        for image_name in os.listdir(class_f):
            image_path = os.path.join(class_f, image_name)
            features = extract_features(image_path)
            test_features.append(features)
            test_target.append(folder)


# In[31]:


predicted = svm_model.predict(test_features)
accuracy = accuracy_score(test_target, predicted)
print("Accuracy: ")
print(accuracy*100)


# In[ ]:




