import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))
import numpy as np
from model_final import unet
import cv2
import os
import h5py

##############################修改路径####################################
path_weights="weights-improvement-43.hdf5"#训练好的权重路径
root_testIn='data/testIn'#测试数据路径（修改）
######################################################################
def load_hdf_test(root1):
    imgf = h5py.File(root1, 'r')
    img_list=[]
    keys = list(imgf.keys())
    for key in keys:
        img_list.append(imgf[str(key)]['data'].value)
    img=np.array(img_list,np.float32) * 0.01

    for i in range(5):
        img[i,:,:] = img[i,:,:] - np.mean(img[i,:,:])
        #img[i, :, :] = (img[i, :, :] - np.mean(img[i,:,:]))/ chan_std[i]
        #img[i,:,:] = img[i,:,:] - b[i] - a[i] * np.sin(img[-1,:,:] / 180. * math.pi)

    img = img[:-1, :, :]
    img=np.transpose(img,[1,2,0])

    return img

model=unet(100,100,5)
model.load_weights(path_weights)
#root1='data/verificationIn'
#root2='data/verificationOut'
#root1='data/testIn'
#root2='data/testOut'
#root1='data/trainIn'
#root2='data/trainOut'
fileList=os.listdir(root_testIn)
print(len(fileList))
i=0
for filename in fileList:
    print(i)
    i +=1
    img = load_hdf_test(root_testIn + '/' + filename)
    #print(img[:,:,1].shape)
    img90 = np.array(np.rot90(img))
    img1 = np.concatenate([img[None], img90[None]])
    img2 = np.array(img1)[:, ::-1]
    img3 = np.array(img1)[:, :, ::-1]
    img4 = np.array(img2)[:, :, ::-1]
    maska = model.predict(img1)
    maskb = model.predict(img2)
    maskc = model.predict(img3)
    maskd = model.predict(img4)

    mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
    mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]
    predicts = mask2 / 8.0
    predicts[predicts>=0.5]=255
    predicts[predicts<0.5]=0
    print(predicts.shape)
    predict=np.squeeze(predicts)
    print(predict.shape)
    output = predict.astype(np.uint8)
    cv2.imwrite('testOut/' + filename[:-4] + '.jpg',output,[int(cv2.IMWRITE_JPEG_QUALITY), 100])




