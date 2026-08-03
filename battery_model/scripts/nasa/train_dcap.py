import sys
import argparse
import os
import numpy as np
import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import json
import time
import keras
from keras import layers
import pickle
from sklearn.preprocessing import StandardScaler
from datetime import datetime as dt

from utils import *
import models

tensorboard_dir = os.environ['TENSORBOARD_DIR']

if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument('--ae-dir', type=str, default="exps/cnn-1/ae")
    args = parser.parse_args()
    print(" ".join(sys.argv))
    
    ae_models_dir = os.path.join(args.ae_dir, "models")
    models_dir = os.path.join(args.out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    train_cellnames, valid_cellnames = read_train_test_cells(args)
    print(train_cellnames, valid_cellnames)
    
    start_time = time.time()
    all_bdfs = get_all_bdfs(train_cellnames + valid_cellnames, args.data_dir)

    X_train, Y_train = extract_dataset(all_bdfs, train_cellnames, args.window_length, args.window_step)
    X_valid, Y_valid = extract_dataset(all_bdfs, valid_cellnames, args.window_length, args.window_step)
    print(f"Data loaded in {time.time() - start_time:.2f} secs")
    
    X_train_scaled = np.zeros(X_train.shape)
    X_valid_scaled = np.zeros(X_valid.shape)
    for i in range(3):
        X_scaler = StandardScaler()
        X_train_scaled[:,:,i] = X_scaler.fit_transform(X_train[:,:,i])
        X_valid_scaled[:,:,i] = X_scaler.transform(X_valid[:,:,i])
        X_scaler_file = os.path.join(models_dir, f"x{i}_scaler.pkl")
        with open(X_scaler_file,'wb') as fp:
            pickle.dump(X_scaler, fp)
        
    Y_scaler = StandardScaler()
    Y_train_scaled = Y_scaler.fit_transform(Y_train.reshape(-1,1)).reshape(-1)
    Y_valid_scaled = Y_scaler.transform(Y_valid.reshape(-1,1)).reshape(-1)
    Y_scaler_file = os.path.join(models_dir, f"y_scaler.pkl")
    with open(Y_scaler_file,'wb') as fp:
        pickle.dump(Y_scaler, fp)
    
    print("train dataset: ", X_train.shape, Y_train.shape)
    print("valid dataset: ", X_valid.shape, Y_valid.shape)
    sys.stdout.flush()
    
    # This is the size of our encoded representations
    encoding_dim = args.latent_size  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
    
    # with open(os.path.join(args.ae_dir, "best-model-epoch.txt"), mode="r") as f:
    #     best_ae_epoch = int(f.readlines()[0].rstrip())

    ae_model_file = os.path.join(ae_models_dir, f"ae.checkpoint.model.keras")
    if os.path.isfile(ae_model_file):
        ae_model = keras.models.load_model(ae_model_file)
        encoder_layers = ae_model.layers[1]
        print(f"loading ae model from {ae_model_file}")
    else:
        encoder_layers = models.get_encoder_layers(encoding_dim, args.window_length)
        print(f"training ae model from scratch")
    
    input_tensor = keras.Input(shape=(args.window_length, 3))
    
    encoded = encoder_layers(input_tensor)
    dcap_layers = models.get_dcap_layers(encoding_dim)
    dcap = dcap_layers(encoded)
    dcap_model = keras.Model(input_tensor, dcap)

    dcap_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=keras.losses.MeanSquaredError(),
    )
    
    timestamp = dt.now().strftime('%Y%m%d-%H%M%S')
    dcap_model.fit(X_train_scaled, Y_train_scaled,
                epochs=args.epochs,
                batch_size=256,
                shuffle=True,
                validation_data=(X_valid_scaled, Y_valid_scaled), 
                callbacks=[
                    keras.callbacks.CSVLogger(os.path.join(args.out_dir, "run_dcap.csv"), separator=",", append=False),
                    keras.callbacks.TensorBoard(log_dir=os.path.join(tensorboard_dir, f"{timestamp}-{args.exp_name}-dcap"), 
                                                update_freq="epoch"),
                    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, 
                                                        patience=20, min_lr=0.0001),
                    keras.callbacks.ModelCheckpoint(os.path.join(models_dir, "dcap.checkpoint.model.keras"), monitor="val_loss", verbose=0, 
                                                        save_best_only=True, save_weights_only=False,
                                                        mode="min", save_freq="epoch", initial_value_threshold=None,),
                    
                    keras.callbacks.ModelCheckpoint(os.path.join(models_dir, "dcap-latest.checkpoint.model.keras"),  
                                                        save_best_only=False, save_weights_only=False, save_freq="epoch"),
                    SaveBestModelEpochCallback(os.path.join(args.out_dir, "best-model-epoch.txt"))
                    ]
    )

    
    print("all done.")