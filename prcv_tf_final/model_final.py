# coding=utf-8
import keras
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D, \
    Activation, Input,GlobalAveragePooling2D,Dense
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.layers.merge import concatenate

def squeeze_excitation_layer(x, out_dim):
    '''
    SE module performs inter-channel weighting.
    '''
    squeeze = GlobalAveragePooling2D()(x)

    excitation = Dense(units=out_dim // 4)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = keras.layers.Reshape((1, 1, out_dim))(excitation)

    scale = keras.layers.multiply([x, excitation])

    return scale

def unet(img_h,img_w,C):
    inputs = Input((img_h, img_w,C))

    conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv1)#100
    conv1 = squeeze_excitation_layer(conv1,64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#50



    conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv2)
    conv2 = squeeze_excitation_layer(conv2, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#25

    conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv3)
    conv3 = squeeze_excitation_layer(conv3,256)

    up6 = concatenate([UpSampling2D(size=(2, 2),interpolation='bilinear')(conv3), conv2], axis=3)#50
    conv6 = Conv2D(128, (3, 3), activation="relu", padding="same")(up6)
    conv6 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv6)
    conv6 = squeeze_excitation_layer(conv6, 128)

    up7 = concatenate([UpSampling2D(size=(2, 2),interpolation='bilinear')(conv6), conv1], axis=3)#100
    conv7 = Conv2D(64, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv7)
    conv7 = squeeze_excitation_layer(conv7, 64)
    conv10 = Conv2D(1, (1, 1),padding="same",activation='sigmoid')(conv7)
    model = Model(inputs=inputs, outputs=conv10)
    adam = keras.optimizers.Adam(lr=0.0001)
    model.summary()
    #parallel_model = multi_gpu_model(model, gpus=2)

    from keras import backend as K


    def combo_dice(y_true, y_pred):
        intersection = K.sum(y_true * y_pred)
        union = K.sum(y_pred * y_pred) + \
                K.sum(y_true * y_true)
        K.set_epsilon(1e-05)
        loss = (intersection + K.epsilon()) / (union + K.epsilon())
        loss_crossentropy = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        loss = loss_crossentropy - K.log(loss)
        return loss


    model.compile(optimizer=adam, loss=[combo_dice], metrics=['binary_accuracy'])
    return model