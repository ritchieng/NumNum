
# coding: utf-8

# ## Deep Convolutional Neural Networks with SVHN

# ### A. Download and Visualize Data

# **Import Libraries**

# In[4]:

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
from scipy import ndimage


# **Create Downloader Function**

# In[8]:

data_folder = 'data/'

def download_data(filename, url, expected_bytes, force=False):
    # Download a file if not present, and make sure it's the right size.
    file_path = data_folder + filename
    if force or not os.path.exists(file_path):
        filename, _ = urlretrieve(url + filename, file_path)
    statinfo = os.stat(file_path)
    if statinfo.st_size == expected_bytes:
        print('Found and verified {}'.format(filename))
    else:
        raise Exception(
          'Failed to verify {}. Can you get to it with a browser?'.format(filename))
    return file_path

def extract_data(filename, force=False):
    # Remove .tar.gz
    root = data_folder + os.path.splitext(os.path.splitext(filename)[0])[0]
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('{} already present - Skipping extraction of {}.'.format(root, filename))
    else:
        print('Extracting data for {}. This may take a while. Please wait.'.format(root))
        tar = tarfile.open(data_folder + filename)
        sys.stdout.flush()
        tar.extractall(data_folder)
        tar.close()
    return root


# **Download Data**

# In[9]:

url = 'http://ufldl.stanford.edu/housenumbers/'
data_folder = 'data/'

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

train_filename = download_data('train.tar.gz', url, 404141560)
test_filename = download_data('test.tar.gz', url, 276555967)
extra_filename = download_data('extra.tar.gz', url, 1955489752)

train_images = extract_data('train.tar.gz')
test_images = extract_data('test.tar.gz')
extra_images = extract_data('extra.tar.gz')


# **Visualize Images from SVHN**

# In[24]:

# Get listing of files in this directory
fn = os.listdir('data/train/')

# **Create Functions to Load Metadata**

# In[40]:

class DigitStructFile:
    def __init__(self, inf):
        self.inf = h5py.File(inf, 'r')
        self.digitStructName = self.inf['digitStruct']['name']
        self.digitStructBbox = self.inf['digitStruct']['bbox']


    def getName(self,n):
        return ''.join([chr(c[0]) for c in self.inf[self.digitStructName[n][0]].value])


    def bboxHelper(self,attr):
        if (len(attr) > 1):
            attr = [self.inf[attr.value[j].item()].value[0][0] for j in range(len(attr))]
        else:
            attr = [attr.value[0][0]]
        return attr


    def getBbox(self,n):
        bbox = {}
        bb = self.digitStructBbox[n].item()
        bbox['height'] = self.bboxHelper(self.inf[bb]["height"])
        bbox['label'] = self.bboxHelper(self.inf[bb]["label"])
        bbox['left'] = self.bboxHelper(self.inf[bb]["left"])
        bbox['top'] = self.bboxHelper(self.inf[bb]["top"])
        bbox['width'] = self.bboxHelper(self.inf[bb]["width"])
        return bbox

    def getDigitStructure(self,n):
        s = self.getBbox(n)
        s['name']=self.getName(n)
        return s
     
    def getAllDigitStructure(self):
        return [self.getDigitStructure(i) for i in range(len(self.digitStructName))]


    def getAllDigitStructure_ByDigit(self):
        pictDat = self.getAllDigitStructure()
        result = []
        structCnt = 1
        for i in range(len(pictDat)):
            item = { 'filename' : pictDat[i]["name"] }
            figures = []
            for j in range(len(pictDat[i]['height'])):
               figure = {}
               figure['height'] = pictDat[i]['height'][j]
               figure['label']  = pictDat[i]['label'][j]
               figure['left']   = pictDat[i]['left'][j]
               figure['top']    = pictDat[i]['top'][j]
               figure['width']  = pictDat[i]['width'][j]
               figures.append(figure)
            structCnt = structCnt + 1
            item['boxes'] = figures
            result.append(item)
        return result


# In[41]:

fin = 'data/test/digitStruct.mat'
dsf = DigitStructFile(fin)
test_data = dsf.getAllDigitStructure_ByDigit()

print('Got test data.')

fin = 'data/train/digitStruct.mat'
dsf = DigitStructFile(fin)
train_data = dsf.getAllDigitStructure_ByDigit()

print('Got train data.')

fin = 'data/extra/digitStruct.mat'
dsf = DigitStructFile(fin)
extra_data = dsf.getAllDigitStructure_ByDigit()

print('Got extra data.')


# ### B. Preprocess Data

# **Create Functions for Processing Images**

# In[42]:

def labels_dict(data_set):
    l_dict = {}
    for item in data_set:
        l_dict[item['filename']] = item['boxes']        
    return l_dict

test_labels_dict = labels_dict(test_data)
train_labels_dict = labels_dict(train_data)
extra_labels_dict = labels_dict(extra_data)


# In[43]:

FLAGS_image_size = 32

def process_images(folder, data_dict):
    
    process_folder = folder + '_processed'
    
    if os.path.exists(process_folder):
        print('{} folder already present - Skipping processingg images in {} folder.'.format(process_folder, folder))
    else:
        os.makedirs(process_folder)
        image_files = os.listdir(folder)
        for image in image_files:
            image_file = os.path.join(folder, image)
            try:
                img = Image.open(image_file)
                image_length, image_height = img.size
                length = len(data_dict[image])
                left, right, top, bottom = [], [], [], []
                for i in range(length):
                    left.append(data_dict[image][i]['left'])
                    right.append(data_dict[image][i]['left'] + data_dict[image][i]['width'])
                    top.append(data_dict[image][i]['top'])
                    bottom.append(data_dict[image][i]['top'] + data_dict[image][i]['height'])
                number_length = max(right)-min(left)
                number_height = min(top)-max(bottom)
                l_crop = np.int32(max([min(left) - .3 * number_length, 0]))
                r_crop = np.int32(min([max(right) + .3 * number_length, image_length]))
                t_crop = np.int32(max([min(top) - .3 * image_height, 0]))
                b_crop = np.int32(min([max(bottom) + .3 * image_height, image_height]))
                img = img.crop([l_crop,t_crop,r_crop,b_crop])
                img = img.resize((FLAGS_image_size, FLAGS_image_size))
                img.save(process_folder+'/'+image, 'PNG') 
            except IOError as e:
                print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
        print('%s folder created.' % process_folder)    
    return


# **Call function for processing images**

# In[44]:

process_images('data/test', test_labels_dict)
process_images('data/train', train_labels_dict)
process_images('data/extra', extra_labels_dict)


# **Create Functions for Loading and Pickling Data**

# In[45]:

# Number of levels per pixel.
FLAGS_pixel_depth = 255.0  
FLAGS_num_labels = 10


def load_data(folder, data_dict, min_num_images):
  
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), FLAGS_image_size, FLAGS_image_size, 1),
                         dtype=np.float32)
    labelset = np.ndarray(shape=(len(image_files), 7))
    
    num_images = 0
    for image in image_files:
        
        image_file = os.path.join(folder, image)
        try:
            image_data = ndimage.imread(image_file).astype(float)   
            image_data = np.dot(image_data[...,:3], [0.299, 0.587, 0.114]).reshape(FLAGS_image_size,FLAGS_image_size,1)
            image_data = (image_data - FLAGS_pixel_depth / 2.0) / FLAGS_pixel_depth
            dataset[num_images, :, :, :] = image_data 
            temp_labels = []
            for i in range(len(data_dict[image])):
                temp_labels.append(data_dict[image][i]['label'])
            labelset[num_images, :] = temp_labels + [-1] * (6-len(data_dict[image])) + [len(data_dict[image])]
            num_images = num_images + 1
        except IOError as e:
            print('Could not process:', image_file, ':', e, '- it\'s ok, skipping.')
    
    dataset = dataset[0:min_num_images, :, :, :]
    labelset = labelset[0:min_num_images, :]
    
    if min_num_images > -1 and num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' % (num_images, min_num_images))
    
    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset, labelset
        

def maybe_pickle(data_folder, label_dict, min_num_images, force=False):
  
    set_image_filename = data_folder + '.pickle'
    set_label_filename = data_folder + '_labels.pickle'
    T = os.path.exists(set_image_filename) and os.path.exists(set_label_filename)
    
    if T and not force:
        print('%s and %s already present - Skipping pickling.' % (set_image_filename, set_label_filename))
    else:
        print('Pickling %s.' % set_image_filename)        
        dataset, labelset = load_data(data_folder, label_dict, min_num_images)
        try:
            with open(set_image_filename, 'wb') as f:
                pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', set_image_filename, ':', e)  
        try:
            with open(set_label_filename, 'wb') as g:
                pickle.dump(labelset, g, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', set_label_filename, ':', e)
  
    return set_image_filename, set_label_filename


# **Call pickling function**

# In[47]:

X_train, y_train = maybe_pickle('data/train_processed', train_labels_dict, -1)
X_test, y_test = maybe_pickle('data/test_processed', test_labels_dict, -1)
X_extra, y_extra = maybe_pickle('data/extra_processed', extra_labels_dict, -1)
