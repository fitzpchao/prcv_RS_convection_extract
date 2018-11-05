# coding=utf-8
import keras
from model_final import unet
from read_data_final import *
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
###########################修改路径##################################
root_trainIn='data/trainIn_1'#训练集输入路径（修改）
root_trainOut='data/trainOut_1'#训练集标签路径（修改）
root_verificationIn='data/verificationIn'#验证集输入路径（修改）
root_verificationOut='data/verificationOut'#验证集标签路径（修改）
path_checkpoints='checkpoints_final1'#模型保存路径
path_log='./final1_log'#log文件保存路径
####################################################################
class Confusion(keras.callbacks.Callback):
    def __init__(self,validation_data,interval=1,score_val=[]):
        self.interval=interval
        self.x_val,self.y_val=validation_data
        self.score_val=score_val
    def on_epoch_end(self,epoch, logs={}):
        if epoch % self.interval == 0:

            pred_y=self.model.predict(self.x_val,verbose=0)
            pred_y[pred_y>=0.5]=1
            pred_y[pred_y<0.5]=0
            true_label=np.reshape(self.y_val,[-1])
            pred_label=np.reshape(pred_y,[-1])
            confused_matrix=confusion_matrix(true_label,pred_label)
            a = confused_matrix[1, 1]
            d = confused_matrix[0, 0]
            b = confused_matrix[0, 1]
            c = confused_matrix[1, 0]
            e = float(((a + b) * (a + c)) / (a + b + c + d))
            ETS = (a - e) / (a + b + c - e)
            s = 0
            if (ETS < 0):
                s = 0
            elif (ETS >= 0.4):
                s = 100
            else:
                s = 625 * ETS * ETS
            self.score_val.append(s)

            np.save('score_val_final.npy',self.score_val)

def getValData():
    fileList=os.listdir(root_verificationIn)
    imgs=[]
    masks=[]
    for filename in fileList:
        img, mask = load_hdf(root_verificationIn + '/' + filename, root_verificationOut + '/' + filename)
        imgs.append(img)
        masks.append(mask)
    imgs=np.array(imgs,np.float32)
    masks=np.array(masks)
    return imgs,masks




def train():
    EPOCHS = 200
    BS = 16
    model = unet(100,100,5)
    fileroot = path_checkpoints
    filepath = os.path.join(fileroot,'weights-improvement-{epoch:02d}.hdf5')
    if(not os.path.exists(fileroot)):
        os.makedirs(fileroot)
    modelcheck = ModelCheckpoint(filepath, monitor='val_acc', mode='max',verbose=1)
    tb_cb=keras.callbacks.TensorBoard(log_dir=path_log)
    X_val, Y_val=getValData()
    score_val=[]
    Conf = Confusion(validation_data=(X_val, Y_val),score_val=score_val)
    #callable = [modelcheck,tb_cb,Conf]
    callable = [modelcheck, tb_cb]
    train_set, val_set = get_train_val(root_trainIn,root_verificationIn)
    train_numb = len(train_set)
    valid_numb = len(val_set)
    print("the number of train data is", train_numb)
    print("the number of val data is", valid_numb)
    model.fit_generator(generator=generateData(BS,root_trainIn,root_trainOut,train_set), steps_per_epoch=train_numb // BS,
                        validation_data=generateValidData(BS,root_verificationIn,root_verificationOut,val_set),validation_steps=valid_numb//BS,
                        epochs=EPOCHS,callbacks=callable, workers=1)
if (__name__ == '__main__'):
    train()


