from keras.layers import Conv3D,ConvLSTM2D,TimeDistributed,Dense,RepeatVector,Concatenate,Input

from keras import Model,Sequential

def conv3d_time(input,repeater):
    repeat = repeater()
    conv3d_o = TimeDistributed(Conv3D(
           filters=20,
           kernel_size=(4,4,4),
           strides=(3,1,1),
           padding='same',
           activation='relu',
           data_format='channels_last'
           ))
    conv3d_out = conv3d_o(input)
    return conv3d_out


def conv3d_convlstm(input_shape,sample_len,timesteps):
    #输入层
    input  = Input(input_shape)
    # 复制层
    repeater = RepeatVector(timesteps)
    # 拼接层
    concat = Concatenate(axis=1)
    # convlstm2d层
    convlstm = ConvLSTM2D(filters=20,
                          kernel_size=(4,4),
                          strides=(1,1),
                          padding='same',
                          activation='relu',
                          return_sequences=True)

    conv3dlayer = Conv3D(filters=3,
                         kernel_size=(4,4,4),
                         strides=(1,1,1),
                         padding='same',
                         activation='relu',
                         )

    # sample_time = list()
    # for _ in range(sample_len):
    #     conv3d_output = conv3d_time(input)
    #     sample_time.append(conv3d_output)
    #
    # concat_output = concat(sample_time)
    # convlstm_output = convlstm(concat_output)
    conv3d_output = conv3d_time(input)
    convlstm_output = convlstm(conv3d_output)
    conv3dlayer_output = conv3dlayer(convlstm_output)

    model = Model(input=input,outputs=conv3dlayer_output)
    return model