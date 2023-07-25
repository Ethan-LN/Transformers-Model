import numpy as np
import tensorflow as tf

class PositionalEncoding(Layer):
    def __init__(self, max_steps, d_model):
        super(PositionalEncoding, self).__init__()
        self.max_steps = max_steps
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(max_steps, d_model)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, max_steps, d_model):
        angle_rads = self.get_angles(
            np.arange(max_steps)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
# Define Transformer model
def create_transformer_model(input_shape, d_model, num_heads, num_layers):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_layers):
        x = MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model, dropout=0.1
        )(x, x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = PositionalEncoding(max_steps=10000, d_model=input_shape[-1])(x)
    x = Dense(input_shape[-1], activation="linear")(x)
    model = Model(inputs=inputs, outputs=x)
    return model