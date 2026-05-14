import sys
import os
import numpy as np
import time
import keras
import pickle
from sklearn.preprocessing import StandardScaler
from datetime import datetime as dt

import models
import utils

tensorboard_dir = os.environ['TENSORBOARD_DIR']

if __name__ == "__main__":
    parser = utils.get_parser()

    args = parser.parse_args()
    print(" ".join(sys.argv))
    utils.save_config_file(args)

    models_dir = os.path.join(args.out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    train_cellnames, valid_cellnames = utils.read_train_test_cells(args)
    print(train_cellnames, valid_cellnames)
    
    start_time = time.time()
    all_bdfs = utils.get_all_bdfs(train_cellnames + valid_cellnames, args.data_dir)

    X_train, Y_train = utils.extract_dataset(all_bdfs, train_cellnames, args.window_length, args.window_step)
    X_valid, Y_valid = utils.extract_dataset(all_bdfs, valid_cellnames, args.window_length, args.window_step)
    print(f"Data loaded in {time.time() - start_time:.2f} secs")
    
    start_time = time.time()
    print("Y_train stats (before scale): ", Y_train.mean(), Y_train.std())
    Y_scaler = StandardScaler(with_mean=False).fit(Y_train.reshape(-1,1))
    Y_train_scaled = Y_scaler.transform(Y_train.reshape(-1,1)).reshape(-1)
    print("Y_train stats (after scale): ", Y_train_scaled.mean(), Y_train_scaled.std())
    
    X_scalers = []
    X_train_scaled = np.zeros(X_train.shape)
    X_valid_scaled = np.zeros(X_valid.shape) 
    for i in range(3):
        X_scaler = StandardScaler(with_mean=False).fit(X_train[:,:,i])
        X_train_scaled[:,:,i] = X_scaler.transform(X_train[:,:,i])
        X_valid_scaled[:,:,i] = X_scaler.transform(X_valid[:,:,i])
        with open(os.path.join(models_dir, f"x{i}_scaler.pkl"),'wb') as fp:
            pickle.dump(X_scaler, fp)
    
    with open(os.path.join(models_dir, "y_scaler.pkl"),'wb') as fp:
        pickle.dump(Y_scaler, fp)
    
    print("train dataset: ", X_train.shape, Y_train.shape)
    print("valid dataset: ", X_valid.shape, Y_valid.shape)
    sys.stdout.flush()
    
    encoding_dim = args.latent_size

    input_tensor = keras.Input(shape=(args.window_length, 3))

    encoder_layers = models.get_encoder_layers(encoding_dim, args.window_length)
    decoder_layers = models.get_decoder_layers(encoding_dim, args.window_length)
    encoded = encoder_layers(input_tensor)
    decoded = decoder_layers(encoded)
    autoencoder = keras.Model(input_tensor, decoded)

    encoder = keras.Model(input_tensor, encoded)

    encoded_input = keras.Input(shape=(encoding_dim,))
    decoded = decoder_layers(encoded_input)

    decoder = keras.Model(encoded_input, decoded)

    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=keras.losses.MeanSquaredError(),
    )
    autoencoder.summary()
    
    timestamp = dt.now().strftime('%Y%m%d-%H%M%S')
    
    autoencoder.fit(X_train_scaled, X_train_scaled,
                epochs=args.epochs,
                batch_size=256,
                shuffle=True,
                validation_data=(X_valid_scaled, X_valid_scaled), 
                callbacks=[
                    keras.callbacks.CSVLogger(os.path.join(args.out_dir, "run_ae.csv"), separator=",", append=False),
                    keras.callbacks.TensorBoard(log_dir=os.path.join(tensorboard_dir, f"{timestamp}-{args.exp_name}-ae"), 
                                                write_graph=True, update_freq="epoch"),
                    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, 
                                                        patience=20, min_lr=0.0001),
                    keras.callbacks.ModelCheckpoint(os.path.join(models_dir, "ae.checkpoint.model.keras"), monitor="val_loss", verbose=0, 
                                                        save_best_only=True, save_weights_only=False,
                                                        mode="min", save_freq="epoch", initial_value_threshold=None,),
                    keras.callbacks.ModelCheckpoint(os.path.join(models_dir, "ae-latest.checkpoint.model.keras"),  
                                                        save_best_only=False, save_weights_only=False, save_freq="epoch"),
                    utils.SaveBestModelEpochCallback(os.path.join(args.out_dir, "best-model-epoch.txt"))
                    ]
                )

    print("all done.")