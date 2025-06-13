import tensorflow as tf

class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim, activation='relu')

    def call(self, x):
        x = self.fc(x)
        return x

# we can use them seperately or as a one code collectively as used in notebook.