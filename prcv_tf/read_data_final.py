# coding=utf-8
import numpy as np
import random
import h5py
import os


def load_hdf(root1,root2):
    imgf = h5py.File(root1, 'r')
    img_list=[]
    mask_list=[]
    keys = list(imgf.keys())
    for key in keys:
        img_list.append(imgf[str(key)]['data'].value)
    maskf = h5py.File(root2, 'r')
    keys = list(maskf.keys())
    for key in keys:
        mask_list.append(maskf[key]['data'].value)
    img=np.array(img_list,np.float32) * 0.01

    for i in range(5):
        img[i,:,:] = img[i,:,:] - np.mean(img[i,:,:])
        #img[i, :, :] = (img[i, :, :] - np.mean(img[i,:,:]))/ chan_std[i]
        #img[i,:,:] = img[i,:,:] - b[i] - a[i] * np.sin(img[-1,:,:] / 180. * math.pi)

    img = img[:-1, :, :]
    mask=np.array(mask_list,np.float32)#shapr=[1,100,100]
    img=np.transpose(img,[1,2,0])
    mask = np.transpose(mask, [1, 2, 0])
    mask[mask>0]=1
    rot_p=random.random()
    flip_p = random.random()
    if(rot_p<0.5):
        pass
    elif(rot_p>=0.5):
        for k in range(5):
            img[:, :, k] = np.rot90(img[:, :, k])
        mask=np.rot90(mask)
    if(flip_p<0.25):
        pass
    elif(flip_p<0.5):
        for k in range(5):
            img[:, :, k] = np.fliplr(img[:, :, k])
        mask=np.fliplr(mask)
    elif (flip_p < 0.75):
        for k in range(5):
            img[:, :, k] = np.flipud(img[:, :, k])
        mask = np.flipud(mask)
    elif (flip_p < 1.0):
        for k in range(5):
            img[:, :, k] = np.fliplr(np.flipud(img[:, :, k]))
        mask = np.fliplr(np.flipud(mask))

    return img,mask

def load_hdf_test(root1,root2):
    imgf = h5py.File(root1, 'r')
    img_list=[]
    mask_list=[]
    keys = list(imgf.keys())
    for key in keys:
        img_list.append(imgf[str(key)]['data'].value)
    maskf = h5py.File(root2, 'r')
    keys = list(maskf.keys())
    for key in keys:
        mask_list.append(maskf[key]['data'].value)
    img=np.array(img_list,np.float32) * 0.01

    for i in range(5):
        img[i,:,:] = img[i,:,:] - np.mean(img[i,:,:])
        #img[i, :, :] = (img[i, :, :] - np.mean(img[i,:,:]))/ chan_std[i]
        #img[i,:,:] = img[i,:,:] - b[i] - a[i] * np.sin(img[-1,:,:] / 180. * math.pi)

    img = img[:-1, :, :]
    mask=np.array(mask_list,np.float32)#shapr=[1,100,100]
    img=np.transpose(img,[1,2,0])
    mask = np.transpose(mask, [1, 2, 0])
    mask[mask>0]=1

    return img,mask

def get_train_val(root1,root2):
    url_train=os.listdir(root1)
    random.shuffle(url_train)
    url_val=os.listdir(root2)
    random.shuffle(url_val)
    return url_train,url_val

def generateData(batch_size,root1,root2,data=[]):
    #print 'generateData...'
    while True:
        train_data = []
        train_label = []
        random.shuffle(data)
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img,label= load_hdf(root1 + '/'+url,root2 + '/'+url)
            train_data.append(img)
            train_label.append(label)

            if batch % batch_size==0:
                #print 'get enough bacth!\n'
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                yield (train_data,train_label)
                train_data = []
                train_label = []
                batch = 0

def generateValidData(batch_size,root1,root2,data=[]):
    #print 'generateValidData...'
    while True:
        valid_data = []
        valid_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img, label = load_hdf(root1 + '/'+url, root2 + '/'+url)
            valid_data.append(img)
            valid_label.append(label)
            if batch % batch_size==0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label)
                yield (valid_data,valid_label)
                valid_data = []
                valid_label = []
                batch = 0