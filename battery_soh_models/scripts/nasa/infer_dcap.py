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
    parser.add_argument('--dcap-dir', type=str, default="exps/cnn-1/dcap")
    args = parser.parse_args()
    print(" ".join(sys.argv))
        
    models_dir = os.path.join(args.out_dir, "models")
    
    train_cellnames, valid_cellnames = read_train_test_cells(args)
    print(train_cellnames, valid_cellnames)
    
    dcap_model = keras.models.load_model(os.path.join(args.dcap_dir, "models", f"dcap.checkpoint.model.keras"))
    dcap_model.summary()

    X_scalers = []
    for i in range(3):
        X_scaler_file = os.path.join(args.dcap_dir, "models", f"x{i}_scaler.pkl")
        with open(X_scaler_file,'rb') as fp:
            X_scaler = pickle.load(fp)
            X_scalers.append(X_scaler)

    Y_scaler_file = os.path.join(args.dcap_dir, "models", f"y_scaler.pkl")
    with open(Y_scaler_file,'rb') as fp:
        Y_scaler = pickle.load(fp)

    loss_fn = keras.losses.MeanSquaredError()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "csv"), exist_ok=True)

    def infer_cells(cellnames, label):
        cap_mpe = []
        all_loss = []
        
        fig = plt.figure(figsize=(12,1.2*len(cellnames)//4+1))
        for i,cellname in tqdm(enumerate(cellnames)):
            bdf = get_cell_bdf(cellname, args.data_dir)
            X_, Y_ = extract_dataset({cellname: bdf}, [cellname], args.window_length, args.window_step)
            Y_scaled = Y_scaler.transform(Y_.reshape(-1, 1)).reshape(-1)
            
            for j in range(3):
                X_[:,:,j] = X_scalers[j].transform(X_[:,:,j])
            
            Y_pred = dcap_model(X_)
            loss = loss_fn(Y_pred, Y_scaled).numpy()
            all_loss.append(loss)
            Y_pred = Y_scaler.inverse_transform(Y_pred).reshape(-1)
            cell_dcap_errors = get_errors(Y_, Y_pred)
            # print(cell_dcap_errors)
            beg_idxs = get_wins_beg_idxs(bdf, args.window_length, args.window_step)
            init_cap = bdf.loc[beg_idxs[0]]["cap"]
            pred_cap = init_cap + np.cumsum(Y_pred)*1e-6
            true_cap = np.interp(beg_idxs, bdf["idx"], bdf["cap"])
            cell_cap_errors = get_errors(true_cap, pred_cap)
            
            ax = fig.add_subplot(int(len(cellnames)/4+1), 4, i+1)
            ax.plot(beg_idxs, true_cap)
            ax.plot(beg_idxs, pred_cap, marker='x', markersize=1)
            ax.text(.01, .01, f'{cellname}: {cell_cap_errors[1]:.2f}', ha='left', va='bottom', transform=ax.transAxes)
        
            cap_mpe.append(cell_cap_errors[1])

            all_info = np.vstack([beg_idxs, Y_pred, Y_, pred_cap, true_cap]).T
            print(all_info.shape)
            column_names = ["start_idx", "pred_dcap", "true_dcap", "pred_cap", "true_cap"]
            all_info = pd.DataFrame(all_info, columns=column_names)
            all_info.to_csv(os.path.join(args.out_dir, "csv", f"pred_{cellname}.csv"))
        mpe_mean = np.array(cap_mpe).mean()
        loss_mean = np.array(all_loss).mean()
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        fig.suptitle(f"{label} cap mpe: {mpe_mean:.3f}\n{label} loss mean: {loss_mean:.3f}")
        print(f"{label} cap mpe: ", mpe_mean)
        print(f"{label} loss mean: ", loss_mean)
        
        plt.savefig(os.path.join(args.out_dir, f"preds_{label}.pdf"), bbox_inches='tight') 

        ############################################################
    infer_cells(sorted(valid_cellnames), "valid")
    infer_cells(sorted(train_cellnames), "train")

    print("all done.")