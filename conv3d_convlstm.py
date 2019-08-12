from keras.models import Sequential,load_model,Model
from keras.layers import Conv3D,Reshape,ConvLSTM2D,Input,Add,Concatenate,BatchNormalization

import numpy as np
from numpy import concatenate
import os
import cv2

def load_image(dir,width,hight):
    imgd = list()
    for img in os.listdir(dir):
        # imgdata = cv2.imdecode(np.fromfile(os.path.join(dir,img),dtype=np.uint8),-1)
        # flags：读入图片的标志
        # cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道
        # cv2.IMREAD_GRAYSCALE：读入灰度图片
        # cv2.IMREAD_UNCHANGED：顾名思义，读入完整图片，包括alpha通道
        imgdata = cv2.imread(os.path.join(dir,img),cv2.IMREAD_UNCHANGED)
        imgdata = imgdata.reshape(width ,hight,imgdata.shape[2])
        imgdata = imgdata.reshape(1,imgdata.shape[0],imgdata.shape[1],imgdata.shape[2])
        imgd.append(imgdata)
    imgd = tuple(imgd)
    imgd = np.concatenate(imgd,axis=0)
    return imgd
# dir = 'D:\大气项目\image_seq_predict\ImagesSequencesPredictions-master\ImagesSequencesPredictions-master\samples\\'


def conv3d_time(input):
    conv3d_o = Conv3D(
           filters=20,
           kernel_size=(7,2,2),
           strides=(7,1,1),
           padding='same',
           activation='relu',
           data_format='channels_last'
           )
    # normal = BatchNormalization()
    conv3d_ou = Conv3D(filters=10,
                       kernel_size=(2,2,2),
                       strides=(1,1,1),
                       padding='same',
                       activation='relu',
                       data_format='channels_last')

    conv3d_out = conv3d_o(input)
    # normal_out = normal(conv3d_out)
    conv3d_output = conv3d_ou(conv3d_out)
    # normal1_out = normal(conv3d_output)

    return conv3d_output


def conv3d_convlstm(input_shape,sample_len):
    #输入层
    input  = Input(input_shape)
    # 拼接层
    normal1 = BatchNormalization()
    concat = Concatenate(axis=1)
    # convlstm2d层
    convlstm = ConvLSTM2D(filters=20,
                          kernel_size=(3,3),
                          strides=(1,1),
                          padding='same',
                          activation='relu',
                          return_sequences=True)

    conv3dlayer = Conv3D(filters=3,
                         kernel_size=(3,3,3),
                         strides=(1,1,1),
                         padding='same',
                         activation='relu',
                         )

    # sample_time = list()
    # for _ in range(sample_len):
    #     conv3d_output = conv3d_time(input)
    #     sample_time.append(conv3d_output)
    # concat_output = concat(sample_time)
    # convlstm_output = convlstm(concat_output)
    conv3d_output = conv3d_time(input)
    convlstm_output = convlstm(conv3d_output)
    # normal_out = normal1(convlstm_output)
    conv3dlayer_output = conv3dlayer(convlstm_output)

    model = Model(input=input,outputs=conv3dlayer_output)
    return model


import  numpy as np
width = 600
hight = 600
sequence = []
# path = os.path.dirname(os.path.abspath(__file__))

from PIL import Image
def load_img(dir,width,hight):
    seq = list()
    for img in os.listdir(dir):
        imgdata = Image.open(os.path.join(dir,img))
        imgdata = imgdata.resize((width,hight),Image.ANTIALIAS)
        imgdata = imgdata.convert('RGB')
        imgdata = np.array(imgdata.getdata()).reshape(1,width,hight,3)
        seq.append(imgdata)
    seq = tuple(seq)
    sequence = np.concatenate(seq,axis=0)
    return sequence

data,label = list(),list()
for _ in range(2,11):
    dir = "D:\image\\{num}\\x\\".format(num=_)
    imgdata = load_img(dir,width=width,hight=hight)
    imgdata = imgdata.reshape(1,imgdata.shape[0],imgdata.shape[1],imgdata.shape[2],imgdata.shape[3])
    data.append(imgdata)

data = tuple(data)
data = np.concatenate(data,axis=0)
print(data.shape)


for _ in range(2,11):
    dir =  "D:\image\\{num}\\y\\".format(num=_)
    imgdata = load_img(dir, width=width, hight=hight)
    imgdata = imgdata.reshape(1, imgdata.shape[0], imgdata.shape[1], imgdata.shape[2], imgdata.shape[3])
    label.append(imgdata)
label = tuple(label)
label = np.concatenate(label,axis=0)
print(label.shape)


# def to_sequence(data,timesteps,sample_len):
#     start = 0
#     X,y = list(),list()
#     for _ in range(timesteps-1,sample_len):
#         # print(data[1])
#         c = data[1,:,:,:,:]
#         print(c.shape[0])
#         data_x = map(lambda x : np.concatenate((data[x,:,:,:,:]),axis=0),[x for x in range(timesteps)])
#         print(data_x)
#         data_x = np.array(list(data_x))
#         X.append(data_x)
#     return np.array(X)
#
# X = to_sequence(data,3,10)
# print(X.shape)



input_shape = (data.shape[1],data.shape[2],data.shape[3],data.shape[4])
model = conv3d_convlstm(input_shape,9)
model.summary()
model.compile(loss='mse',optimizer='adam',metrics=['mse','acc'])
model.fit(data,label,batch_size=2,epochs=20)
model.save('conv3d_convlstm_rgb_7_7.h5')

from  keras.models import load_model
model = load_model('conv3d_convlstm_rgb_7_7.h5')
test = data[-1:]
pre = model.predict(test)
print(pre)
print(pre.shape)
pre = pre.reshape(pre.shape[2],pre.shape[3],pre.shape[4])

pre = Image.fromarray(np.uint8(pre),mode='RGB')
pre.save('pre9.png')
# pre = pre.reshape(pre.shape[0]*pre.shape[1]*pre.shape[2]*pre.shape[3]*pre.shape[4],1)
# y = label[-1:]
# y  =  y.reshape(y.shape[0]*y.shape[1]*y.shape[2]*y.shape[3]*y.shape[4],1)
# mae = np.mean(np.abs(pre-y))
# max_mae = np.max(np.abs(pre-y))
# print(mae)
# print(max_mae)

