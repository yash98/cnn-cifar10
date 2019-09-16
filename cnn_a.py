import sys
import time
import numpy as np
import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping

# from google.colab import drive
# drive.mount('/content/drive')

slice_size = 5000
num_classes = 10

# python cnn_a.py train.csv test.csv outputfile
time_start = time.time()
inp = np.loadtxt(sys.argv[1])
# train_x = [np.reshape(image, (32, 32, 3)) for image in inp[:, :-1]]
x_train = np.reshape(inp[slice_size:, :-1],
                     (inp.shape[0]-slice_size, 32, 32, 3))
y_train = k.utils.to_categorical(inp[slice_size:, -1:], num_classes)
x_test = np.reshape(inp[:slice_size, :-1],
                    (slice_size, 32, 32, 3))
y_test = k.utils.to_categorical(inp[:slice_size, -1:], num_classes)
inp = np.loadtxt(sys.argv[2])
x_out = np.reshape(inp[:, :-1], (inp.shape[0], 32, 32, 3))
del(inp)

epochs = 10000
batch_size = 32*32
# time_max = 5*60

k.backend.tensorflow_backend._get_available_gpus()

model = Sequential()

# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(64, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(BatchNormalization())
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

checkpointer = ModelCheckpoint(filepath="bestweights.hdf5",
                               monitor='val_loss', verbose=1, save_best_only=True,
                               save_weights_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          callbacks=[checkpointer, earlystopper])

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

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

model.load_weights("bestweights.hdf5")
y_out = model.predict_classes(x_out)
np.savetxt(sys.argv[3], y_out)
