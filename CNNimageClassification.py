#!/usr/bin/env python
# coding: utf-8

# In[104]:


from tensorflow.keras.models import Sequential


# In[106]:


from tensorflow.keras.layers import Conv2D , MaxPooling2D , Dense , Flatten


# In[108]:


import numpy as np
import matplotlib.pyplot as plt
import random


# # LOAD DATASET

# In[110]:


x_train_labels = np.loadtxt(r"C:\Users\Admin\Downloads\labels.csv",delimiter = ",")
x_train = np.loadtxt(r"C:\Users\Admin\Downloads\input.csv",delimiter = ",")
x_test = np.loadtxt(r"C:\Users\Admin\Downloads\input_test.csv",delimiter = ",")
x_test_labels = np.loadtxt(r"C:\Users\Admin\Downloads\labels_test.csv",delimiter = ",")


# # Converting the shape to 100 , 100 , 3      
# 3 denotes the rgb colours 

# In[112]:


x_train = x_train.reshape(len(x_train) , 100 , 100 , 3 )
x_test = x_test.reshape(len(x_test) , 100 , 100 , 3 )


# In[114]:


print(x_train.shape)
print(x_test.shape)


# # Preprocessing the Data 

# In[116]:


x_train = x_train / 255.0
x_test = x_test / 255.0


# In[118]:


x_train


# In[122]:


plt.imshow(x_test[0])
plt.show()


# # making our model 

# In[166]:


model = Sequential([
    Conv2D(32 ,(3,3) , activation = "relu" , input_shape = (100,100,3)),
    MaxPooling2D((2,2)),

    Conv2D(32 ,(3,3) , activation = "relu" ),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(64 , activation = "relu"),
    Dense(1 , activation = "sigmoid"),

])

    


# # Compiling 

# In[136]:


model.compile(loss = "binary_crossentropy", optimizer = "adam" , metrics = ["accuracy"])


# # Training our Model

# In[144]:


model.fit(x_train,x_train_labels,epochs = 5,batch_size = 64)


# In[146]:


model.evaluate(x_test,x_test_labels)


# In[164]:


y_pred = model.predict(x_test[0].reshape(1,100,100,3))
print(y)
plt.imshow(x_test[0])
plt.show()


# In[ ]:




