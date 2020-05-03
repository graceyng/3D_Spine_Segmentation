from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Lambda, Reshape
from tensorflow.keras.layers import Conv2D, UpSampling2D, SpatialDropout2D, MaxPooling2D, ZeroPadding2D, Conv2DTranspose
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, SpatialDropout3D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Permute
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import os
import numpy as np



#=================================================================================
# CODE FOR 3D MODEL
#=================================================================================


########### /////////////
###### CHANNELS FIRST SETUP
########### ///////////////

def conv_block_simple_3D(prevlayer, filters, prefix, initializer="he_normal", strides=(1, 1, 1)):
    conv = Conv3D(filters, (3, 3, 3), padding="same", kernel_initializer=initializer, strides=strides, name=prefix + "_conv",
         data_format='channels_first')(prevlayer)
         
    conv = BatchNormalization(name=prefix + "_bn",
        axis=1)(conv)

    conv = Activation('relu', name=prefix + "_activation")(conv)

    return conv



# def conv_block_simple_3D(prevlayer, filters, prefix, initializer="he_normal", strides=(1, 1, 1)):
#     conv = Conv3D(filters, (3, 3, 3), padding="same", kernel_initializer=initializer, strides=strides, name=prefix + "_conv",
#          data_format='channels_first')(prevlayer)
#     conv = BatchNormalization(name=prefix + "_bn",
#         axis=1)(conv)
#     conv = Activation('relu', name=prefix + "_activation")(conv)
#     return conv



def conv_block_simple_no_bn_3D(prevlayer, filters, prefix, initializer="he_normal", strides=(1, 1, 1)):
    conv = Conv3D(filters, (3, 3, 3), padding="same", kernel_initializer=initializer, strides=strides, name=prefix + "_conv",
        data_format='channels_first')(prevlayer)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv


########### /////////////
###### CHANNELS LAST SETUP
########### ///////////////


# def conv_block_simple_3D(prevlayer, filters, prefix, strides=(1, 1, 1)):
#     conv = Conv3D(filters, (3, 3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv",
#             data_format='channels_last')(prevlayer)
#     conv = BatchNormalization(name=prefix + "_bn",  axis=4 )(conv)
#     conv = Activation('relu', name=prefix + "_activation")(conv)
#     return conv

# def conv_block_simple_no_bn_3D(prevlayer, filters, prefix, strides=(1, 1, 1)):
#     conv = Conv3D(filters, (3, 3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv",
#         data_format='channels_last')(prevlayer)
#     conv = Activation('relu', name=prefix + "_activation")(conv)
#     return conv


def get_3d_unet_9layers(input_shape):

    img_input = Input(input_shape)

    conv1 = conv_block_simple_3D(img_input, 32, "conv1_1")
    conv1 = conv_block_simple_3D(conv1, 32, "conv1_2")
    pool1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same", name="pool1",
        data_format='channels_first')(conv1)

    conv2 = conv_block_simple_3D(pool1, 64, "conv2_1")
    conv2 = conv_block_simple_3D(conv2, 64, "conv2_2")
    pool2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same", name="pool2",
        data_format='channels_first')(conv2)

    conv3 = conv_block_simple_3D(pool2, 128, "conv3_1")
    conv3 = conv_block_simple_3D(conv3, 128, "conv3_2")
    pool3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same", name="pool3",
        data_format='channels_first')(conv3)

    conv4 = conv_block_simple_3D(pool3, 256, "conv4_1")
    conv4 = conv_block_simple_3D(conv4, 256, "conv4_2")
    pool4 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same", name="pool4",
        data_format='channels_first')(conv4)

    conv5 = conv_block_simple_3D(pool4, 512, "conv5_1")
    conv5 = conv_block_simple_3D(conv5, 512, "conv5_2")
    conv5 = conv_block_simple_3D(conv5, 512, "conv5_3")

    up6 = concatenate([UpSampling3D(data_format='channels_first')(conv5), conv4], axis=1)
    conv6 = conv_block_simple_3D(up6, 256, "conv6_1")
    conv6 = conv_block_simple_3D(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling3D(data_format='channels_first')(conv6), conv3], axis=1)
    conv7 = conv_block_simple_3D(up7, 128, "conv7_1")
    conv7 = conv_block_simple_3D(conv7, 128, "conv7_2")

    up8 = concatenate([UpSampling3D(data_format='channels_first')(conv7), conv2], axis=1)
    conv8 = conv_block_simple_3D(up8, 64, "conv8_1")
    conv8 = conv_block_simple_3D(conv8, 64, "conv8_2")

    up9 = concatenate([UpSampling3D(data_format='channels_first')(conv8), conv1], axis=1)
    conv9 = conv_block_simple_3D(up9, 32, "conv9_1")
    conv9 = conv_block_simple_3D(conv9, 32, "conv9_2")

    conv9 = SpatialDropout3D(rate=0.1,data_format='channels_first')(conv9)

    prediction = Conv3D(1, (1, 1, 1), activation="sigmoid", name="prediction",data_format='channels_first')(conv9)
    model = Model(img_input, prediction)

    return model



def get_simple_3d_unet(input_shape):

    img_input = Input(input_shape)
    conv1 = conv_block_simple_3D(img_input, 32, "conv1_1")
    conv1 = conv_block_simple_3D(conv1, 32, "conv1_2")
    pool1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same", name="pool1",
        data_format='channels_first')(conv1)

    conv2 = conv_block_simple_3D(pool1, 64, "conv2_1")
    conv2 = conv_block_simple_3D(conv2, 64, "conv2_2")
    pool2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same", name="pool2",
        data_format='channels_first')(conv2)

    conv3 = conv_block_simple_3D(pool2, 128, "conv3_1")
    conv3 = conv_block_simple_3D(conv3, 128, "conv3_2")
    pool3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same", name="pool3",
        data_format='channels_first')(conv3)

    conv4 = conv_block_simple_3D(pool3, 256, "conv4_1")
    conv4 = conv_block_simple_3D(conv4, 256, "conv4_2")
    conv4 = conv_block_simple_3D(conv4, 256, "conv4_3")

    up5 = concatenate([UpSampling3D(data_format='channels_first')(conv4), conv3], axis=1)
    conv5 = conv_block_simple_3D(up5, 128, "conv5_1")
    conv5 = conv_block_simple_3D(conv5, 128, "conv5_2")

    up6 = concatenate([UpSampling3D(data_format='channels_first')(conv5), conv2], axis=1)
    conv6 = conv_block_simple_3D(up6, 64, "conv6_1")
    conv6 = conv_block_simple_3D(conv6, 64, "conv6_2")

    up7 = concatenate([UpSampling3D(data_format='channels_first')(conv6), conv1], axis=1)
    conv7 = conv_block_simple_3D(up7, 32, "conv7_1")
    conv7 = conv_block_simple_3D(conv7, 32, "conv7_2")

    conv7 = SpatialDropout3D(rate=0.2,data_format='channels_first')(conv7)

    prediction = Conv3D(1, (1, 1, 1), activation="sigmoid", name="prediction",data_format='channels_first')(conv7)
    model = Model(img_input, prediction)
    return model

def get_shallow_3d_unet(input_shape):


    img_input = Input(input_shape)
    conv1 = conv_block_simple_3D(img_input, 32, "conv1_1")
    conv1 = conv_block_simple_3D(conv1, 32, "conv1_2")
    pool1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same", name="pool1",
            data_format='channels_first')(conv1)

    conv2 = conv_block_simple_3D(pool1, 64, "conv2_1")
    conv2 = conv_block_simple_3D(conv2, 64, "conv2_2")
    pool2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same", name="pool2",
            data_format='channels_first')(conv2)

    conv3 = conv_block_simple_3D(pool2, 128, "conv3_1")
    conv3 = conv_block_simple_3D(conv3, 128, "conv3_2")
    pool3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same", name="pool3",
            data_format='channels_first')(conv3)


    up4 = concatenate([UpSampling3D(data_format='channels_first')(conv3), conv2], axis=1) 
    conv4 = conv_block_simple_3D(up4, 64, "conv4_1")
    conv4 = conv_block_simple_3D(conv4, 64, "conv4_2")

    up5 = concatenate([UpSampling3D(data_format='channels_first')(conv4), conv1], axis=1)
    conv5 = conv_block_simple_3D(up5, 32, "conv5_1")
    conv5 = conv_block_simple_3D(conv5, 32, "conv5_2")

    conv5 = SpatialDropout3D(rate=0.2,data_format='channels_first')(conv5)

    prediction = Conv3D(1, (1, 1, 1), activation="sigmoid", name="prediction",
         data_format='channels_first')(conv5)


    model = Model(img_input, prediction)
    return model



def get_small_3d_unet(input_shape):
    img_input = Input(input_shape)
    conv1 = conv_block_simple_3D(img_input, 32, "conv1_1")
    conv1 = conv_block_simple_3D(conv1, 32, "conv1_2")
    pool1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same", name="pool1",
            data_format='channels_first')(conv1)

    conv2 = conv_block_simple_3D(pool1, 64, "conv2_1")
    conv2 = conv_block_simple_3D(conv2, 64, "conv2_2")
    conv2 = conv_block_simple_3D(conv2, 64, "conv2_3")

    up3 = concatenate([UpSampling3D(data_format='channels_first')(conv2), conv1 ], axis=1 )
    conv3 = conv_block_simple_3D( up3, 32, "conv3_1")
    conv3 = conv_block_simple_3D( conv3, 32, "conv3_2")

    conv3 = SpatialDropout3D(rate=0.2,data_format='channels_first')(conv3)

    prediction = Conv3D(1, (1, 1, 1), activation="sigmoid", name="prediction", 
         data_format='channels_first')(conv3)

    model = Model(img_input, prediction)
    return model




def get_test_3d_small_cnn(input_shape):
    img_input = Input(input_shape)
    conv1 = conv_block_simple_3D( img_input, 32, "conv1" )
    conv2 = conv_block_simple_3D( conv1, 16, "conv2"     )
    conv3 = conv_block_simple_3D( conv2, 8, "conv3"     )


    # conv3 = SpatialDropout3D(rate=0.2,data_format='channels_first')(conv3)

    prediction = Conv3D(1, (1, 1, 1), activation="sigmoid", name="prediction", 
         data_format='channels_first')(conv3)

    model = Model(img_input, prediction)

    return model





































    # # 256 * 256 * 60
    # img_input = Input(input_shape)
    # conv1 = conv_block_simple_3D(img_input, 32, "conv1_1")
    # conv1 = conv_block_simple_3D(conv1, 32, "conv1_2")
    # pool1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same", name="pool1",
    #         data_format='channels_last')(conv1)

    # conv2 = conv_block_simple_3D(pool1, 64, "conv2_1")
    # conv2 = conv_block_simple_3D(conv2, 64, "conv2_2")
    # pool2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same", name="pool2",
    #         data_format='channels_last')(conv2)

    # conv3 = conv_block_simple_3D(pool2, 128, "conv3_1")
    # conv3 = conv_block_simple_3D(conv3, 128, "conv3_2")
    # pool3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same", name="pool3",
    #         data_format='channels_last')(conv3)

    # conv4 = conv_block_simple_3D(pool3, 256, "conv4_1")
    # conv4 = conv_block_simple_3D(conv4, 256, "conv4_2")
    # conv4 = conv_block_simple_3D(conv4, 256, "conv4_3")

    # print('conv1')
    # print(conv1)
    # print('conv2')
    # print(conv2)

    # print('conv3')
    # print(conv3)
    # print('conv4')
    # print(conv4)
    # print()
    # print()
    # print()



    # # ERROR HERE
    # # up5 = concatenate([UpSampling3D(size=(2,2,1.875))(conv4), conv3], axis=1)
    # # Tell it to upsample3D conv 4 to shape 64 64 16
    # # 2 2 1.875

    # up5 = concatenate([UpSampling3D()(conv4), conv3], axis=4)
 
    # # up5 = concatenate([UpSampling3D()(conv4), conv3], axis=1)
    # conv5 = conv_block_simple_3D(up5, 128, "conv5_1")
    # conv5 = conv_block_simple_3D(conv5, 128, "conv5_2")

    # up6 = concatenate([UpSampling3D()(conv5), conv2], axis=4)
    # conv6 = conv_block_simple_3D(up6, 64, "conv6_1")
    # conv6 = conv_block_simple_3D(conv6, 64, "conv6_2")

    # up7 = concatenate([UpSampling3D()(conv6), conv1], axis=4)
    # conv7 = conv_block_simple_3D(up7, 32, "conv7_1")
    # conv7 = conv_block_simple_3D(conv7, 32, "conv7_2")

    # conv7 = SpatialDropout3D(0.2)(conv7)

    # prediction = Conv3D(1, (1, 1, 1), activation="sigmoid", name="prediction")(conv7)
    # model = Model(img_input, prediction)
    # return model




# def get_simple_3d_unet(input_shape):

#     # 256 * 256 * 60
#     img_input = Input(input_shape)
#     conv1 = conv_block_simple_3D(img_input, 32, "conv1_1")
#     conv1 = conv_block_simple_3D(conv1, 32, "conv1_2")
#     pool1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same", name="pool1")(conv1)

#     conv2 = conv_block_simple_3D(pool1, 64, "conv2_1")
#     conv2 = conv_block_simple_3D(conv2, 64, "conv2_2")
#     pool2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same", name="pool2")(conv2)

#     conv3 = conv_block_simple_3D(pool2, 128, "conv3_1")
#     conv3 = conv_block_simple_3D(conv3, 128, "conv3_2")
#     pool3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same", name="pool3")(conv3)

#     conv4 = conv_block_simple_3D(pool3, 256, "conv4_1")
#     conv4 = conv_block_simple_3D(conv4, 256, "conv4_2")
#     conv4 = conv_block_simple_3D(conv4, 256, "conv4_3")

#     print('conv1')
#     print(conv1)
#     print('conv2')
#     print(conv2)

#     print('conv3')
#     print(conv3)
#     print('conv4')
#     print(conv4)
#     print()
#     print()
#     print()



#     # ERROR HERE
#     # up5 = concatenate([UpSampling3D(size=(2,2,1.875))(conv4), conv3], axis=1)
#     # Tell it to upsample3D conv 4 to shape 64 64 16
#     # 2 2 1.875

#     up5 = concatenate([UpSampling3D()(conv4), conv3], axis=1)
 
#     # up5 = concatenate([UpSampling3D()(conv4), conv3], axis=1)
#     conv5 = conv_block_simple_3D(up5, 128, "conv5_1")
#     conv5 = conv_block_simple_3D(conv5, 128, "conv5_2")

#     up6 = concatenate([UpSampling3D()(conv5), conv2], axis=1)
#     conv6 = conv_block_simple_3D(up6, 64, "conv6_1")
#     conv6 = conv_block_simple_3D(conv6, 64, "conv6_2")

#     up7 = concatenate([UpSampling3D()(conv6), conv1], axis=1)
#     conv7 = conv_block_simple_3D(up7, 32, "conv7_1")
#     conv7 = conv_block_simple_3D(conv7, 32, "conv7_2")

#     conv7 = SpatialDropout3D(0.2)(conv7)

#     prediction = Conv3D(1, (1, 1, 1), activation="sigmoid", name="prediction")(conv7)
#     model = Model(img_input, prediction)
#     return model


#=================================================================================
# CODE FOR 2D MODEL
#=================================================================================
def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

def conv_block_simple_no_bn(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

# input shape of form (256, 256, 1)
def get_simple_unet(input_shape):
    img_input = Input((input_shape + (1,)))
    conv1 = conv_block_simple(img_input, 32, "conv1_1")
    conv1 = conv_block_simple(conv1, 32, "conv1_2")
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool1")(conv1)

    conv2 = conv_block_simple(pool1, 64, "conv2_1")
    conv2 = conv_block_simple(conv2, 64, "conv2_2")
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool2")(conv2)

    conv3 = conv_block_simple(pool2, 128, "conv3_1")
    conv3 = conv_block_simple(conv3, 128, "conv3_2")
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool3")(conv3)

    conv4 = conv_block_simple(pool3, 256, "conv4_1")
    conv4 = conv_block_simple(conv4, 256, "conv4_2")
    conv4 = conv_block_simple(conv4, 256, "conv4_3")

    up5 = concatenate([UpSampling2D()(conv4), conv3], axis=3)
    conv5 = conv_block_simple(up5, 128, "conv5_1")
    conv5 = conv_block_simple(conv5, 128, "conv5_2")

    up6 = concatenate([UpSampling2D()(conv5), conv2], axis=3)
    conv6 = conv_block_simple(up6, 64, "conv6_1")
    conv6 = conv_block_simple(conv6, 64, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv1], axis=3)
    conv7 = conv_block_simple(up7, 32, "conv7_1")
    conv7 = conv_block_simple(conv7, 32, "conv7_2")

    conv7 = SpatialDropout2D(0.2)(conv7)

    prediction = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv7)
    model = Model(img_input, prediction)
    return model

#=================================================================================
# CODE FOR PBP 2D Model
#           "Paco Bravo Project"
#=================================================================================

def get_simple_unet_pbp(input_shape):
    img_input = Input((1, input_shape, input_shape))
    conv1 = conv_block_simple(img_input, 32, "conv1_1")
    conv1 = conv_block_simple(conv1, 32, "conv1_2")
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool1")(conv1)

    conv2 = conv_block_simple(pool1, 64, "conv2_1")
    conv2 = conv_block_simple(conv2, 64, "conv2_2")
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool2")(conv2)

    conv3 = conv_block_simple(pool2, 128, "conv3_1")
    conv3 = conv_block_simple(conv3, 128, "conv3_2")
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool3")(conv3)

    conv4 = conv_block_simple(pool3, 256, "conv4_1")
    conv4 = conv_block_simple(conv4, 256, "conv4_2")
    conv4 = conv_block_simple(conv4, 256, "conv4_3")

    up5 = concatenate([UpSampling2D()(conv4), conv3], axis=1)
    conv5 = conv_block_simple(up5, 128, "conv5_1")
    conv5 = conv_block_simple(conv5, 128, "conv5_2")

    up6 = concatenate([UpSampling2D()(conv5), conv2], axis=1)
    conv6 = conv_block_simple(up6, 64, "conv6_1")
    conv6 = conv_block_simple(conv6, 64, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv1], axis=1)
    conv7 = conv_block_simple(up7, 32, "conv7_1")
    conv7 = conv_block_simple(conv7, 32, "conv7_2")

    conv7 = SpatialDropout2D(0.2)(conv7)

    prediction = Conv2D(1, (1, 1), activation="relu", name="prediction")(conv7)
    model = Model(img_input, prediction)
    return model

#=================================================================================
# VGG models
#=================================================================================
# input shape of form (224, 224, 1)
def getVGGModel(num_classes, input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1), padding='same',input_shape=(input_shape + (1,)), data_format="channels_last"))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    return model
def getVGGModel2(num_classes,input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1), padding='same',input_shape=input_shape, data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    return model
def getVGGModel3DLocator(numClasses=2, inputShape=(224,224), numFramesPerStack=8):
    model = Sequential()
    model.add(Conv3D(64, (3, 3, 3), activation='relu', strides=(1,1,1), padding='same',input_shape=inputShape + (numFramesPerStack, 1), data_format="channels_last"))
    model.add(Conv3D(64, (3, 3, 3), activation='relu', strides=(1,1,1), padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 1), strides=(2,2,1)))

    model.add(Conv3D(128, (3, 3, 3), activation='relu', strides=(1,1,1), padding='same'))
    model.add(Conv3D(128, (3, 3, 3), activation='relu', strides=(1,1,1), padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 1), strides=(2,2,2)))

    model.add(Conv3D(256, (3, 3, 3), activation='relu', strides=(1,1,1), padding='same'))
    model.add(Conv3D(256, (3, 3, 3), activation='relu', strides=(1,1,1), padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 1), strides=(2,2,1)))

    model.add(Conv3D(512, (3, 3, 3), activation='relu', strides=(1,1,1), padding='same'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu', strides=(1,1,1), padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 1), strides=(2,2,1)))

    model.add(Conv3D(512, (3, 3, 3), activation='relu', strides=(1,1,1)))
    model.add(Conv3D(512, (3, 3, 1), activation='relu', strides=(1,1,1), padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2,2,2)))
    
    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    
    
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(numClasses, activation="softmax"))
    return model