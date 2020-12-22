#!/usr/bin/env python
# coding: utf-8

# In[1]:


data_dir = 'dataset/'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os


# In[6]:


df_data = pd.DataFrame()
df_labels = []

print('Reading data ...')
for fileName in glob.glob(os.path.join('enhancer_test', '*.{}'.format('csv'))):
    df = pd.read_csv(fileName, header=None)
#     print('In processing: ', fileName)
    df_new = df.stack().to_frame().T
    df_data = df_data.append(df_new, ignore_index=True)
    df_labels.append(1)


# In[7]:


df_data


# In[8]:


len(df_labels)


# In[9]:


print('Reading negative data ...')
for fileName in glob.glob(os.path.join('non_cv', '*.{}'.format('csv'))):
    df = pd.read_csv(fileName, header=None)
#     print('In processing: ', fileName)
    df_new = df.stack().to_frame().T
    df_data = df_data.append(df_new, ignore_index=True)
    df_labels.append(0)


# In[19]:


df_data.shape


# ### Separating to training and validation data

# In[22]:


X_trn = df_data
y_trn = df_labels


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_trn, y_trn, test_size=0.25, random_state=42)


# ## Deep learning implementation

# In[32]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, ZeroPadding1D, MaxPooling1D, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import utils
from tensorflow.keras import optimizers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix


# In[20]:


num_features = 153600

nb_classes = 2
nb_kernels = 3
nb_pools = 2
nb_epochs = 50


# In[21]:


def _1D_CNN_model():
    model = Sequential()

    model.add(Conv1D(32, 3, input_shape=(num_features,1), activation='relu'))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(2))

    # model.add(Conv1D(128, 3, activation='relu'))
    # model.add(MaxPooling1D(2))

    # model.add(Conv1D(256, 3, activation='relu'))
    # model.add(MaxPooling1D(2))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[30]:


def _2D_CNN_model():
    model = Sequential()

    model.add(Conv2D(32, 3, 3, input_shape=(768,200,1), activation='relu'))
    model.add(MaxPooling2D(2))

    model.add(Conv1D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(2))

    # model.add(Conv1D(128, 3, activation='relu'))
    # model.add(MaxPooling1D(2))

    # model.add(Conv1D(256, 3, activation='relu'))
    # model.add(MaxPooling1D(2))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[27]:


import matplotlib.pyplot as plt

model = _1D_CNN_model()

# Plot model history
history = model.fit(np.asarray(X_train).reshape(len(np.asarray(X_train)),num_features,1), utils.to_categorical(y_train,nb_classes), 
                    validation_data=(np.asarray(X_test).reshape(len(np.asarray(X_test)),num_features,1), utils.to_categorical(y_test,nb_classes)),
                    epochs=nb_epochs, batch_size=16, verbose=1)


# In[29]:


# Plot training & validation accuracy values
fig = plt.figure()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[ ]:


_2D_model = _2D_CNN_model()

# Plot model history
_2D_history = _2D_model.fit(np.asarray(X_train).reshape(len(np.asarray(X_train)),768,200,1), utils.to_categorical(y_train,nb_classes), 
                    validation_data=(np.asarray(X_test).reshape(len(np.asarray(X_test)),768,200,1), utils.to_categorical(y_test,nb_classes)),
                    epochs=nb_epochs, batch_size=16, verbose=1)


# In[ ]:




