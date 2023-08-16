import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Flatten
from keras.utils import to_categorical
from keras.optimizers import SGD

(X,Y),(x,y)=mnist.load_data()

X=X.reshape(X.shape[0],28,28,1).astype("float32")
x=x.reshape(x.shape[0],28,28,1).astype("float32")

X=X/255
x=x/255

Y=to_categorical(Y)
y=to_categorical(y)

labelcnt=Y.shape[1]

with tf.device("/GPU:0"):
    model=Sequential()

model.add(
    Conv2D(
        filters=16,
        kernel_size=(3,3),
        input_shape=(28,28,1),
        activation="relu",
        kernel_initializer="he_normal"
    )
)
model.add(BatchNormalization())
model.add(
    MaxPool2D(
        pool_size=(2,2)
    )
)
model.add(
    Conv2D(
        filters=32,
        kernel_size=(4,4),
        activation="relu",
        kernel_initializer="he_normal"
    )
)
model.add(BatchNormalization())
model.add(
    MaxPool2D(
        pool_size=(2,2)
    )
)
model.add(Flatten())
model.add(
    Dense(
        units=labelcnt,
        activation="softmax",
        kernel_initializer="he_normal"
    )
)
model.compile(
    optimizer=SGD(0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
    
model.fit(X,Y,validation_data=(x,y),epochs=100,batch_size=32,verbose=2)
