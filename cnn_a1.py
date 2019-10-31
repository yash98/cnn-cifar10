import sys
import numpy as np
import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import adam

# from google.colab import drive
# drive.mount('/content/drive')

# slice_size = 5000
num_classes = 10

# python cnn_a.py train.csv test.csv outputfile
inp = np.loadtxt(sys.argv[1])
# train_x = [np.reshape(image, (32, 32, 3)) for image in inp[:, :-1]]
x_train = np.reshape(inp[:, :-1],
                     (inp.shape[0], 32, 32, 3))
y_train = k.utils.to_categorical(inp[:, -1:], num_classes)
# x_train = np.reshape(inp[slice_size:, :-1],
#                      (inp.shape[0]-slice_size, 32, 32, 3))
# y_train = k.utils.to_categorical(inp[slice_size:, -1:], num_classes)
# x_test = np.reshape(inp[:slice_size, :-1],
#                     (slice_size, 32, 32, 3))
# y_test = k.utils.to_categorical(inp[:slice_size, -1:], num_classes)
inp = np.loadtxt(sys.argv[2])
x_out = np.reshape(inp[:, :-1], (inp.shape[0], 32, 32, 3))
del(inp)

epochs = 50
batch_size = 128
# time_max = 5*60

k.backend.tensorflow_backend._get_available_gpus()

model = Sequential()

# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
                 input_shape=x_train.shape[1:],
                 kernel_initializer='he_normal', bias_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
                 kernel_initializer='he_normal', bias_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu',
                kernel_initializer='he_normal', bias_initializer='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu',
                kernel_initializer='he_normal', bias_initializer='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu',
                kernel_initializer='he_normal', bias_initializer='he_normal'))
model.add(Dense(num_classes,
                kernel_initializer='he_normal', bias_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('softmax'))

opt = adam()
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255.0
# x_test /= 255.0

checkpointer = ModelCheckpoint(filepath="bestweights_a.hdf5",
                               monitor='val_acc', verbose=1, save_best_only=True,
                               save_weights_only=True)
earlystopper = EarlyStopping(monitor='val_acc', patience=5, verbose=1)

callbacks = [checkpointer, earlystopper]
# callbacks = [checkpointer]

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1,
          callbacks=callbacks)

# time_passed = time.time() - time_start
# time_left = time_max - time_passed
# epoch_num = 0
# time_epoch_avg = 0
# time_epoch_sum = 0
# while(time_left > 2*time_epoch_avg):
#     time_epoch_start = time.time()
#     epoch_num += 1
#     print("epoch_num: ", epoch_num)
#     model.fit(x_train, y_train,
#               batch_size=batch_size,
#               epochs=1,
#               validation_data=(x_test, y_test),
#               )
#     time_passed = time.time()-time_start
#     time_left = time_max - time_passed
#     time_epoch_end = time.time()
#     time_passed_epoch = time_epoch_end - time_epoch_start
#     time_epoch_sum += time_passed_epoch
#     time_epoch_avg = time_epoch_sum/epoch_num
#     print("time_left: ", time_left)
#     print("time_epoch_avg: ", time_epoch_avg)
# print("time_epoc")

model.load_weights("bestweights_a.hdf5")

# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])

y_out = np.argmax(model.predict(x_out), axis=1)
np.savetxt(sys.argv[3], y_out, fmt='%i')
