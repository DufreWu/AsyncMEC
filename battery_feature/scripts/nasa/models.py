import keras
from keras import layers

n_pool_layers = 1
n_channels = 8
filter_size = 8

def get_encoder_layers(encoding_dim, window_length):    
    flatten_size = n_channels * window_length // pow(2, n_pool_layers)
    return keras.Sequential(
            [
                keras.Input(shape=(window_length, 3,)),
                layers.Conv1D(n_channels, (filter_size), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.AveragePooling1D(2, padding='same'),
                layers.Flatten(),
                layers.Dense(encoding_dim, activation='relu')
            ], name="encoder"
        )

def get_decoder_layers(encoding_dim, window_length):
    flatten_size = n_channels * window_length // pow(2, n_pool_layers)
    return keras.Sequential(
            [
                keras.Input(shape=(encoding_dim,)),
                layers.Dense(flatten_size, activation='relu'),
                layers.Reshape((flatten_size // n_channels, n_channels)),
                layers.Conv1D(n_channels, (filter_size), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.UpSampling1D(2),
                layers.Conv1D(3, (filter_size), activation=None, padding='same'),
                
            ], name="decoder"
        )

def get_dcap_layers(encoding_dim):
    return keras.Sequential(
            [
                keras.Input(shape=(encoding_dim,)),
                layers.Dropout(0.4),
                layers.BatchNormalization(),
                layers.Dense(1, activation=None)
            ], name="dcap"
        )