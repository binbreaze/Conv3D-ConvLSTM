import numpy as np
from PIL import Image
import os


width = 600
hight = 600
def load_img(dir,width,hight):
    seq = list()
    for img in os.listdir(dir):
        imgdata = Image.open(os.path.join(dir,img))
        imgdata = imgdata.resize((width,hight),Image.ANTIALIAS)
        imgdata = imgdata.convert('RGB')
        imgdata = np.array(imgdata.getdata()).reshape(width,hight,3)
        seq.append(imgdata)
    seq = tuple(seq)
    sequence = np.concatenate(seq,axis=-1)
    return sequence

data,label = list(),list()
for _ in range(2,11):
    dir = "D:\image\\{num}\\x\\".format(num=_)
    imgdata = load_img(dir,width=width,hight=hight)
    imgdata = imgdata.reshape(1,imgdata.shape[0],imgdata.shape[1],imgdata.shape[2])
    data.append(imgdata)

data = tuple(data)
data = np.concatenate(data,axis=0)
print(data.shape)


for _ in range(2,11):
    dir =  "D:\image\\{num}\\y\\".format(num=_)
    imgdata = load_img(dir, width=width, hight=hight)
    imgdata = imgdata.reshape(1, imgdata.shape[0], imgdata.shape[1], imgdata.shape[2])
    label.append(imgdata)
label = tuple(label)
label = np.concatenate(label,axis=0)
print(label.shape)

def to_sequence(data,label,n_input,n_output):
    start =0
    X , y = list(), list()
    for _ in range(len(data)):
        n_end = start + n_input
        n_out = n_end + n_output
        if n_out < len(data):
            X.append(data[start:n_end])
            y.append(label[n_end:n_out])
            start += 1
    return np.array(X),np.array(y)

data_x ,data_y = to_sequence(data,label,2,1)
print(data_x.shape,data_y.shape)
from keras import Model
from keras.layers import ConvLSTM2D,TimeDistributed,Dense,RepeatVector,LSTM,Input,BatchNormalization,Conv3D


def convlstm2d(input_shape):
    input = Input(input_shape)

    convlstmcell = ConvLSTM2D(filters=10,
                              kernel_size=(3,3),
                              strides=(1,1),
                              padding='same',
                              return_sequences=True,
                              data_format='channels_last',
                              activation='relu',
                              )

    # normal = BatchNormalization()

    convlstmcell1 = ConvLSTM2D(filters=10,
                               kernel_size=(3,3),
                               strides=(1,1),
                               padding='same',
                               activation='relu',
                               return_sequences=True,
                               data_format='channels_last')

    # normal1 = BatchNormalization()

    conv3d = Conv3D(filters=3,
                    kernel_size=(2,3,3),
                    strides=(2,1,1),padding='same',
                    activation='relu')

    convlstm_out = convlstmcell(input)

    # normal_out = normal(convlstm_out)

    convlstm1_out = convlstmcell1(convlstm_out)

    # normal1_out = normal1(convlstm1_out)

    conv3d_out = conv3d(convlstm1_out)

    model = Model(input=input,output=conv3d_out)
    return model

input_shape = (None,data_x.shape[2],data_x.shape[3],data_x.shape[4])
model = convlstm2d(input_shape)
model.summary()
model.compile(loss='mse',optimizer='adam',metrics=['mae','mape','acc'])
test_x = data_x[2:4]
test_y = data_y[2:4]
history = model.fit(data_x,data_y,epochs=10,batch_size=2,validation_data=(test_x,test_y))
model.save('convlstm_0807.h5')

from keras.models import load_model
model = load_model('convlstm_0807.h5')
pre = model.predict(data_x[-1:])
print(pre.shape)
pre = pre.reshape(pre.shape[2],pre.shape[3],pre.shape[4])
pre = Image.fromarray(np.uint(pre),mode='RGB')
pre.save('conlstm1.jpg')