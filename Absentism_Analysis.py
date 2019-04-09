#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Make sure to install all these libraries
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics


# In[2]:


Absent_work = pd.read_excel("https://s3-ap-southeast-1.amazonaws.com/edwisor-india-bucket/projects/data/DataN0101/Absenteeism_at_work_Project.xls")


# In[3]:


Absent_work


# In[4]:


Absent_work.describe() # We can see that all the attributes are treated as integers. Which is not desired


# In[5]:


# Further checking the data type of the attributes
Absent_work.info(verbose = True)


# In[6]:


# Cleaning the dataset
Absent_work.columns


# In[7]:


Absent_work.drop(['ID'], axis = 1,inplace = True)
Absent_work.columns


# In[8]:


#### Preprocessing the dataset#######
#Changing attributes to factors as they are taken as integer in python
# Checking NA values. And removing the missing value rows
Absent_work.isnull().sum()


# In[9]:


# Dropping the missing values
Absent_work.dropna(inplace=True)
data_svm = Absent_work
Absent_work.isnull().sum()


# In[10]:


# Changing the attribtues to categorical
Absent_work['Reason for absence']  = Absent_work['Reason for absence'].astype('category')
Absent_work['Seasons']  = Absent_work['Seasons'].astype('category')
Absent_work['Day of the week']  = Absent_work['Day of the week'].astype('category')
Absent_work['Disciplinary failure']  = Absent_work['Disciplinary failure'].astype('category')
Absent_work['Social drinker']  = Absent_work['Social drinker'].astype('category')
Absent_work['Social smoker']  = Absent_work['Social smoker'].astype('category')

Absent_work.dtypes


# In[11]:


get_ipython().magic('matplotlib inline')
Absent_work.hist(column=None, by=None, grid=True, xlabelsize=None, xrot=None, ylabelsize=None, yrot=None, ax=None, sharex=False, sharey=False, figsize=(20,20), layout=(5,3), bins=10)


# In[12]:


get_ipython().magic('matplotlib inline')
boxplot = Absent_work.boxplot(column=['Month of absence', 
       'Transportation expense', 'Distance from Residence to Work',
       'Service time', 'Age', 'Work load Average/day ', 'Hit target', 'Education', 'Son', 'Pet', 'Weight', 'Height', 'Body mass index',
       'Absenteeism time in hours'], return_type='axes', figsize = (30,30))
boxplot


# In[13]:


##### Training the model#####
# Making the train and test split
from sklearn.model_selection import train_test_split
train, test = train_test_split(Absent_work, test_size=0.2)


# In[14]:


print("Length of train set: "+str(len(train)))
print("Length of test set: " +str(len(test)))


# In[15]:


# Training set creation
train_x = train.drop(['Absenteeism time in hours'], axis=1)
train_y = train['Absenteeism time in hours']

# Test set creation

test_x = test.drop(['Absenteeism time in hours'], axis=1)
test_y = test['Absenteeism time in hours']


# In[16]:


print("column name of train dataset:  ", train_x.columns, "\n \n column name of the test dataset: " ,test_x.columns)


# In[17]:


from sklearn.tree import DecisionTreeRegressor 


# In[18]:


d_tree = DecisionTreeRegressor(random_state = 0)


# In[19]:


model = d_tree.fit(train_x, train_y)


# In[20]:


#Prediction
predict_value = model.predict(test_x)
predict_value


# In[21]:


# Calculate RMSE for the predicted values
print((((predict_value - test_y)**2).mean())**(1/2))


# In[22]:


import os


# In[23]:


os.getcwd()


# In[24]:


# SVM Regression
train_svm, test_svm = train_test_split(data_svm, test_size=0.2)

print("Length of train set: "+str(len(train_svm)))
print("Length of test set: " +str(len(test_svm)))


# In[29]:


# Training set creation
#train_svm_y =  train_svm['Absenteeism time in hours']
train_svm_x = train_svm.drop(['Absenteeism time in hours'], axis=1)
# train_svm_y =  train_svm['Absenteeism time in hours']

train_svm_y = train_svm.iloc[:,-1]

# train_svm_y = train_svm_y.to_frame()
# train_svm_y = train_svm_y.astype('int')
# #type(train_svm_x)
train_svm_y.dtypes

# train_svm_y = train_svm_y.astype('int')

# # Test set creation

test_svm_x = test_svm.drop(['Absenteeism time in hours'], axis=1)
test_svm_y = test_svm['Absenteeism time in hours']


# In[28]:


from sklearn.svm import SVR

svm_model = SVR(gamma=0.0001, C=1.0, epsilon=0.1)
model_svm = svm_model.fit(train_svm_x, train_svm_y)
model_svm


# In[31]:


predict_value_svm = svm_model.predict(test_svm_x)
predict_value


# In[32]:


# Calculate RMSE for the predicted values
print((((predict_value_svm - test_svm_y)**2).mean())**(1/2))

