import os
import sys
import tarfile
import pickle
import matplotlib.pyplot as plt
import numpy as np, h5py 
import pandas as pd
from six.moves.urllib.request import urlretrieve
from IPython.display import display, Image
from PIL import Image
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, merge
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.callbacks import ModelCheckpoint



# **Open training data**

# In[5]:

with open('data/train_processed.pickle', 'rb') as f:
    X_train = pickle.load(f)
    print('Training data shape:', X_train.shape)
    
with open('data/train_processed_labels.pickle', 'rb') as f:
    y_train = pickle.load(f)
    print('Training label shape:',y_train.shape)


# **Open extra data**

# In[6]:

with open('data/extra_processed.pickle', 'rb') as f:
    X_extra = pickle.load(f)
    print('Extra data shape:', X_extra.shape)
    
with open('data/extra_processed_labels.pickle', 'rb') as f:
    y_extra = pickle.load(f)
    print('Extra label shape:', y_extra.shape)


# **Open test data**

# In[7]:

with open('data/test_processed.pickle', 'rb') as f:
    X_test = pickle.load(f)
    print('Test data shape:', X_test.shape)
    
with open('data/test_processed_labels.pickle', 'rb') as f:
    y_test = pickle.load(f)
    print('Test label shape:', y_test.shape)


# **Create larger training set by merging test and extra datasets**

# In[8]:

X_train = np.concatenate([X_train, X_extra])
y_train = np.concatenate([y_train, y_extra])


# **Create validation and training datasets from merged datasets**

# In[9]:

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state=8)


# **Modify labels for missing digits**

# In[10]:

def process_labels(labels):
    np.place(labels,labels==10.0, 0.0)
    np.place(labels,labels==-1.0, 10.0)
    return

for i in [y_train, y_val, y_test]:
    process_labels(i)



# **One-hot-encode labels** 

# In[12]:

num_labels = 10
def one_hot_encode(y):
     return (np.arange(num_labels) == y[:,None]).astype(np.float32).reshape(-1, 5, 10)


# In[13]:

y_train_dummy = one_hot_encode(y_train[:,:5].reshape(-1,))
y_val_dummy = one_hot_encode(y_val[:,:5].reshape(-1,))
y_test_dummy = one_hot_encode(y_test[:,:5].reshape(-1,))



# **Reproducibility**

# In[15]:

np.random.seed(8)


# **ConvNet**


# In[19]:

y_train_dummy_1 = y_train_dummy[:, 0, :]
y_train_dummy_2 = y_train_dummy[:, 1, :]
y_train_dummy_3 = y_train_dummy[:, 2, :]
y_train_dummy_4 = y_train_dummy[:, 3, :]
y_train_dummy_5 = y_train_dummy[:, 4, :]


# In[20]:

y_val_dummy_1 = y_val_dummy[:, 0, :]
y_val_dummy_2 = y_val_dummy[:, 1, :]
y_val_dummy_3 = y_val_dummy[:, 2, :]
y_val_dummy_4 = y_val_dummy[:, 3, :]
y_val_dummy_5 = y_val_dummy[:, 4, :]


# In[21]:

y_test_dummy_1 = y_test_dummy[:, 0, :]
y_test_dummy_2 = y_test_dummy[:, 1, :]
y_test_dummy_3 = y_test_dummy[:, 2, :]
y_test_dummy_4 = y_test_dummy[:, 3, :]
y_test_dummy_5 = y_test_dummy[:, 4, :]


# In[22]:

y_train_lst = [y_train_dummy_1, y_train_dummy_2, y_train_dummy_3, y_train_dummy_4, y_train_dummy_5]
y_val_lst = [y_val_dummy_1, y_val_dummy_2, y_val_dummy_3, y_val_dummy_4, y_val_dummy_5]
y_test_lst = [y_test_dummy_1, y_test_dummy_2, y_test_dummy_3, y_test_dummy_4, y_test_dummy_5]


# In[23]:

batch_size = 32
nb_classes = 10
nb_epoch = 20

# Image Dimensions
_, img_rows, img_cols, img_channels = X_train.shape


# In[24]:

model_input = Input(shape=(img_rows, img_cols, img_channels))

x = Convolution2D(32, 3, 3, border_mode='same')(model_input)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Convolution2D(32, 3, 3, border_mode='same')(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Flatten()(x)

x = Dense(1024, activation="relu")(x)

# length = Dense(4, activation='softmax')(x)
digit_1 = Dense(nb_classes, activation='sigmoid')(x)
digit_2 = Dense(nb_classes, activation='sigmoid')(x)
digit_3 = Dense(nb_classes, activation='sigmoid')(x)
digit_4 = Dense(nb_classes, activation='sigmoid')(x)
digit_5 = Dense(nb_classes, activation='sigmoid')(x)

branches = [digit_1, digit_2, digit_3, digit_4, digit_5]

model = Model(input=model_input, output=branches)


# In[25]:

# Load weights
# model.load_weights("weights/trial_1_weights.h5")


# In[26]:

# SGD 
# learning_rate = 0.01
# decay_rate = learning_rate / nb_epoch
# momentum = 0.9

# sgd = SGD(lr=learning_rate, decay=decay_rate, momentum=momentum, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])


# In[27]:

# Checkpoint
filepath='weights/trial_1_weights.best.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True,
    mode='min')
callbacks_list = [checkpoint]


# In[28]:

history = model.fit(X_train, y_train_lst, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=callbacks_list, verbose=1, validation_data=(X_val, y_val_lst))


# **Save Weights and Model with Serialization**

# Trial 1

# In[29]:

# Trial 1: Serialize model to JSON
model_json = model.to_json()
with open('weights/trial_1_model.json', 'w') as json_file:
    json_file.write(model_json)

# Trial 1: Serialize weights to HDF5
model.save_weights('weights/trial_1_weights.h5')
print("Saved model to disk")


# **Validation Set Overall Accuracy**


# In[ ]:

y_pred_val = model.predict(X_val, verbose=1)


# In[34]:

# List to narray
arr = np.array(y_pred_val)  

arr[arr < 0.5] = 0            
arr[arr >= 0.5] = 1

# narray to list
y_pred_val_2 = arr.tolist()   


# In[39]:

val_predictions = np.array(y_pred_val_2)
val_labels = np.array(y_val_lst)

def accuracy(predictions, labels):
    return (predictions == labels).all(axis=0).all(axis=-1).mean()

print('Validation accuracy', accuracy(val_predictions, val_labels))



score = model.evaluate(X_val, y_val_lst, verbose=1)
print('Validation error:', score)


# **Test Set Error Score**

# In[215]:

y_pred_test = model.predict(X_test, verbose=1)

# List to narray
arr = np.array(y_pred_val)  

arr[arr < 0.5] = 0            
arr[arr >= 0.5] = 1

# narray to list
y_pred_test_2 = arr.tolist()   

test_predictions = np.array(y_pred_test_2)
test_labels = np.array(y_test_lst)

print('Test accuracy', accuracy(test_predictions, test_labels))