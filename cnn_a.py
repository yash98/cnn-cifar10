import sys
import numpy as np
import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

num_classes = 10
epochs = 100
batch_size = 5000

if __name__ == "__main__":
    # python cnn_a.py train.csv test.csv outputfile
    inp = np.loadtxt(sys.argv[1])
    # train_x = [np.reshape(image, (32, 32, 3)) for image in inp[:, :-1]]
    x_train = np.reshape(inp[batch_size:, :-1], (inp.shape[0], 32, 32, 3))
    y_train = k.utils.to_categorical(inp[batch_size:, -1:], num_classes)
    x_test = np.reshape(inp[:batch_size, :-1], (inp.shape[0], 32, 32, 3))
    y_test = k.utils.to_categorical(inp[:batch_size, -1:], num_classes)
    inp = np.loadtxt(sys.argv[2])
    x_out = np.reshape(inp[:, :-1], (inp.shape[0], 32, 32, 3))
    del(inp)

    model = Sequential()

    model = Sequential()
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(64, (3, 3),
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(ZeroPadding2D(padding=(1, 1)))
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
    model.add(Activation('softmax'))

    opt = k.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              )

    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
