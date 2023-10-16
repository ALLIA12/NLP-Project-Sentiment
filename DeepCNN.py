import tensorflow as tf
from tensorflow.keras import layers

class DeepCNN(tf.keras.Model):
  def __init__(self,
               vocab_size,
               emb_dim=128,
               nb_filters=50,
               FFN_units=512,
               nb_classes=2,
               dropout_rate=0.1,
               training=False,
               name="DeepCNN",
               ):
    super(DeepCNN,self).__init__(name=name)
    self.embedding= layers.Embedding(vocab_size,
                                      emb_dim)
    self.bigram = layers.Conv1D(filters=nb_filters,
                                kernel_size=2,
                                padding="valid",
                                activation="relu")
    self.pool_1= layers.GlobalMaxPool1D()
    self.trigram = layers.Conv1D(filters=nb_filters,
                                kernel_size=3,
                                padding="valid",
                                activation="relu")
    self.pool_2= layers.GlobalMaxPool1D()
    self.quadgram = layers.Conv1D(filters=nb_filters,
                                kernel_size=4,
                                padding="valid",
                                activation="relu")
    self.pool_3= layers.GlobalMaxPool1D()
    self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
    self.dropout = layers.Dropout(rate = dropout_rate)
    if nb_classes == 2:
      self.last_dense = layers.Dense(units=1,activation="sigmoid")
    else:
      # You will need to get the maximum probablity later
      self.last_dense = layers.Dense(units=nb_classes,activation="softmax")

  def call(self,inputs,training):
    x = self.embedding(inputs)
    x_1 = self.bigram(x)
    x_1 = self.pool_1(x_1)
    x_2 = self.trigram(x)
    x_2 = self.pool_2(x_2)
    x_3 = self.quadgram(x)
    x_3 = self.pool_3(x_3)

    merged = tf.concat([x_1,x_2,x_3],axis=-1) # something like this (batchsize, 3*nb_filters)
    merged = self.dense_1(merged)
    merged = self.dropout(merged,training)
    output = self.last_dense(merged)
    return output

