import pandas as pd
import os
import numpy as np
import cv2

def create_csv_file(path):
    folders = []
    filenames = os.listdir(path)
    for folder in filenames:
        folders.append(folder)

    fruits = {
        "Apple" : 0,
        "Banana" : 1,
        "Grape" : 2,
        "Mango" : 3,
        "Strawberry" : 4
    }

    img_list_train = []
    label_list_train = []
    img_list_valid = []
    label_list_valid = []
    for type in folders:
        filenames = os.listdir(path + "/" + type)
        for classes in filenames:
            images = os.listdir(path+ "/" + type + "/" + classes)
            for img in images:
                if img.endswith(".jpeg"):
                    if type == "train":
                        img_list_train.append(path+ "/" + type + "/" + classes + "/" + img)
                        label_list_train.append(fruits[classes])
                    if type == "valid": 
                        img_list_valid.append(path+ "/" + type + "/" + classes + "/" + img)
                        label_list_valid.append(fruits[classes])


    df = pd.DataFrame({'Img_path':img_list_train , 'label': label_list_train})
    df2 = pd.DataFrame({'Img_path':img_list_valid , 'label': label_list_valid})


    df.to_csv("fruits_train.csv", index=False)
    df2.to_csv("fruits_valid.csv", index=False)
    


def create_mean_and_std_for_images(csv_file):
    df = pd.read_csv(csv_file)

    mean = np.array([0.,0.,0.])
    stdTemp = np.array([0.,0.,0.])
    std = np.array([0.,0.,0.])

    numSamples = len(df)

    for i in range(0, len(df)):

        im = cv2.imread(df.iloc[i,0])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(float) / 255.
    
        for j in range(3):
            mean[j] += np.mean(im[:,:,j])

    mean = (mean / numSamples)

    for i in range(0, len(df)):

        im = cv2.imread(df.iloc[i, 0])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(float) / 255. 

        for j in range(3):
            stdTemp[j] += ((im[:,:,j] - mean[j])**2).sum()/(im.shape[0]*im.shape[1])
        
    
    std = np.sqrt(stdTemp/ numSamples)

    return mean, std




#print(create_mean_and_std_for_images('./fruits_train.csv'))
#create_csv_file("./dataset/Fruits Classification"))