import config
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, Dense, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
#Lenet


def build_model_autotune(hp):
    print("[INFO] importing lenet model for autotuning")
    model = Sequential()
    # first set 20
    model.add(Conv2D(hp.Int("conv_1", min_value=8, max_value=32, step=2),
                     (5, 5), padding="same", input_shape=config.INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # second set 50
    model.add(Conv2D(hp.Int("conv_2", min_value=32, max_value=64, step=2),
                    (5, 5), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # full connected 500
    model.add(Flatten())
    model.add(Dense(hp.Int("dense_units", min_value=256, max_value=512, step=64)))
    model.add(Activation("relu"))
    # softmax classifier
    model.add(Dense(config.NUM_CLASSES))
    model.add(Activation("softmax"))
    # return the model

    lr = hp.Choice("learning_rate",
                   # values=[1e-1, 1e-2, 1e-3])
                   values=[1e-1, 1e-2, 1e-3, 1e-4])
    opt = Adam(learning_rate=lr)

    # compile the model
    model.compile(optimizer=opt, loss="binary_crossentropy",
                  metrics=["accuracy"])


    return model
