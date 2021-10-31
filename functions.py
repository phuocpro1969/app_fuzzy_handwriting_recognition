import pandas as pd
import cv2
import numpy as np
import os 
import pickle

PATH=os.getcwd()
DATASET_PATH=os.path.join(PATH, 'datasets')
WIDTH=16
HEIGHT=16
K=WIDTH*HEIGHT

def join_path(path, name):
    return os.path.join(path, name)

def readTrainedFile():
    try:
        vecs=pickle.load(open(join_path(PATH, 'model/model.h5'), 'rb'))
        return vecs
    except:
        return None

def saveTrainedFile(dict):
    pickle.dump(dict, open(join_path(PATH, 'model/model.h5'), 'wb'))


def readDataset():
    dict_img = {}
    
    df = pd.read_csv(join_path(DATASET_PATH, "english.csv"))
    all_links=df['image']
    all_labels=df['label']
    labels=all_labels.unique()
    
    for label in labels:
        dict_img[label] = []
    
    for i in range(len(all_links)):
        dict_img[all_labels[i]].append(all_links[i])
    return dict_img

def readImage(path):
    return cv2.imread(path, 0)

def resizeImage(image):
    return cv2.resize(image, (HEIGHT, WIDTH), interpolation=cv2.INTER_AREA)

def processImage(image):
    _, image=cv2.threshold(image, 165, 255, cv2.THRESH_BINARY_INV)
    image=crop(image)
    image=resizeImage(image)
    return image

def clean(vec):
    vec = np.where(vec <= 0.5, 2*vec, vec) 
    vec = np.where(0.5 < vec, 2*vec-1, vec)
    return vec

def readImagesTrain():
    dict_imgs = {}
    for label, values in readDataset().items():
        vec=np.zeros((HEIGHT, WIDTH))
        for value in values:
            image_before=readImage(join_path(DATASET_PATH, value))
            vec_img=processImage(image_before)
            vec+=vec_img
        dict_imgs[label]=clean(vec/len(values))
        
    return dict_imgs

def calc_deltas(dict_vecs, image):
    dict_delta = {}
    for key, vec in dict_vecs.items():
        recruit=vec+image-vec*image
        delta=recruit.sum()/K
        dict_delta[key]=delta
    return dict_delta

def crop(image):
    # crop left    
    while True:
        sum=np.sum(image[:,0])
        if sum != 0:
            break
        else:
            image=image[:, 1:]  
    
    # crop top
    while True:
        sum=np.sum(image[0,:])
        if sum != 0:
            break
        else:
            image=image[1:, :]  
    
    (height, width) = image.shape
    height-=1
    width-=1
    
    # crop right    
    while True:
        width-=1
        sum=np.sum(image[:,width])
        if sum != 0:
            break
        else:
            image=image[:, :width]  
    
    # crop bottom
    while True:
        height-=1
        sum=np.sum(image[height,:])
        if sum != 0:
            break
        else:
            image=image[:height, :]  
    return image