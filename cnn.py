import tensorflow as tf 
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical 
import matplotlib.pyplot as plt 

# conv_layer= tf.keras.layers.Conv2D(
#     filters=32, kernel_size=(3, 3), strides=(1,1), padding="valid", activation= "relu", kernel_initializer="glorot_uniform")

# max_pooling_layer=tf.keras.layers.MaxPool2D(
#     pool_size= (2, 2), strides= None, padding="valid", data_format=None )

# avg_pooling_layer=tf.keras.layers.AveragePooling2D(
#     pool_size= (2, 2), strides= None, padding="valid", data_format=None )

# fully_connected_layer= tf.keras.layers.Dense(
#     units=128, activation= "relu", kernel_initializer="glorot_uniform")


# model= models.Sequential([])

(train_images, train_labels), (test_images, test_labels)=cifar10.load_data()
train_images, test_images = train_images/ 255.0, test_images/ 255.0
train_labels= to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)
model= models.Sequential([ 
    layers.Conv2D(32, (3, 3), activation= "relu", input_shape= (32, 32, 3)), 
    layers.MaxPooling2D(pool_size=(2, 2)), 
    layers.Flatten(),
    layers.Dense(128, activation="relu"), 
    layers.Dense(10, activation="softmax")
])
# adam an optimizer that uses math like a gradient vector to optimize the vector
#it prioritizes metrics (accuracy in this case)
model.compile(optimizer="adam", 
              loss= "categorical_crossentropy", 
              metrics= ["accuracy"])
#epochs are like practice tests, dont put too many or else it will memorize the answers asnd not learn anything and FAIL
#too few is also problem and not practicing enough
#validation data is what you are testing 
#train label=practice problem, ttestdata= practice test 
history= model.fit (train_images, train_labels, epochs= 10, batch_size =64, validation_data=(test_images, test_labels))