from keras.models import Model
from tensorflow import keras
from keras.layers import add, Conv2D, MaxPooling2D, Input, concatenate, UpSampling2D, \
  BatchNormalization, Activation, SeparableConv2D, Conv2DTranspose


def InputLayer(layerin, Filters=0):
  layerout = Conv2D(Filters, (3, 3), strides=2, padding='same')(layerin)
  layerout = BatchNormalization()(layerout)
  layerout = Activation('relu')(layerout)
  return layerout

def DownSampling(layerin, Filters=0):
  # The number of Filters = Filters
  layerout = Activation('relu')(layerin)
  layerout = SeparableConv2D(Filters, (3, 3), padding='same')(layerout)
  layerout = BatchNormalization()(layerout)

  layerout = Activation('relu')(layerout)
  layerout = SeparableConv2D(Filters, (3, 3), padding='same')(layerout)
  layerout = BatchNormalization()(layerout)

  layerout = MaxPooling2D((3, 3), strides=2, padding='same')(layerout)
  residual = Conv2D(Filters, (1, 1), strides=2, padding='same')(layerin)
  layerout = add([layerout, residual])
  return layerout

def UpSampling(layerin, Filters=0):
  # The number of Filters = Filters
  layerout = Activation('relu')(layerin)
  layerout = Conv2DTranspose(Filters, (3, 3), padding='same')(layerout)
  layerout = BatchNormalization()(layerout)

  layerout = Activation('relu')(layerout)
  layerout = Conv2DTranspose(Filters, (3, 3), padding='same')(layerout)
  layerout = BatchNormalization()(layerout)

  layerout = UpSampling2D(2)(layerout)
  residual = UpSampling2D(2)(layerin)
  residual = Conv2D(Filters, (1, 1), padding='same')(residual)
  layerout = add([layerout, residual])
  return layerout


## Building Unet model
def Build(Input_dimension):
  # Input Layer
  Img_width, Img_height = Input_dimension
  Channel_in = Input(shape=(Img_width, Img_height, 1))
  layer1 = InputLayer(Channel_in, Filters=32)

  # DownSampling
  layer2 = DownSampling(layer1, Filters=64)
  layer3 = DownSampling(layer2, Filters=128)
  layer4 = DownSampling(layer3, Filters=256)

  # UpSampling
  layer5 = UpSampling(layer4, Filters=256)
  layer6 = UpSampling(layer5, Filters=128)
  layer7 = UpSampling(layer6, Filters=64)
  layer8 = UpSampling(layer7, Filters=32)

  # Output Layer
  Channel_out = Conv2D(2, (3, 3), activation='softmax', padding='same')(layer8)
  return Model(Channel_in, Channel_out)


def Compiler(model):
  return model.comiple(optimizer='rmsprop', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

def Fit(model, X, Y, BatchSize=0, Epochs=0, ValidationSplit=0):
  return model.fit(X, Y, batch_size=BatchSize, epochs=Epochs, validation_split=ValidationSplit)

def Saver(model, name):
  return model.save(name)