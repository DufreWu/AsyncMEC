import argparse
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import json
import keras
import pickle


class SaveBestModelEpochCallback(keras.callbacks.Callback):
    def __init__(self, outfile_path):
        super().__init__()
        self.best_loss = np.inf
        self.outfile_path = outfile_path
    def on_epoch_end(self, epoch, logs=None):
        curr_loss = logs["val_loss"]
        if curr_loss < self.best_loss:
            with open(self.outfile_path, mode="w") as f:
                print(epoch, file=f)
                print(f"best model found in epoch {epoch}")
            self.best_loss = curr_loss


class SaveBestCapacityMAPEModelEpochCallback(keras.callbacks.Callback):
    def __init__(self, valid_data_dict, Y_scaler, window_length, outfile_path, outmodel_path, validate_every=10):
        super().__init__()
        self.best_mape = np.inf
        self.valid_data_dict = valid_data_dict
        self.Y_scaler = Y_scaler
        self.window_length = window_length
        self.outfile_path = outfile_path
        self.outmodel_path = outmodel_path
        self.validate_every = validate_every
        
        with open(self.outfile_path, mode="a") as f:
            print(f"epoch,cap_mape_mean,val_loss", file=f)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.validate_every == 0:
            cap_mapes = []
            for cellname in self.valid_data_dict.keys():
                X_, Y_, X_scaled, Y_scaled, bdf = self.valid_data_dict[cellname]
                Y_pred = self.model(X_scaled)
                Y_pred = self.Y_scaler.inverse_transform(Y_pred).reshape(-1)

                beg_idxs = get_wins_beg_idxs(bdf, self.window_length, self.window_length)
                init_cap = bdf.loc[beg_idxs[0]]["cap"]
                pred_cap = init_cap + np.cumsum(Y_pred)*1e-6
                true_cap = np.interp(beg_idxs, bdf["idx"], bdf["cap"])
                cell_cap_errors = get_errors(true_cap, pred_cap)
                cap_mapes.append(cell_cap_errors)
            cap_mapes = np.array(cap_mapes)
            cap_mape_mean = cap_mapes[:,1].mean()
            
            print()
            print(f"cap mape on epoch {epoch}: {cap_mape_mean}")
            with open(self.outfile_path, mode="a") as f:
                print(f"{epoch},{cap_mape_mean},{logs['val_loss']}", file=f)

            if cap_mape_mean < self.best_mape:
                self.best_mape = cap_mape_mean
                self.model.save(self.outmodel_path, overwrite=True)
                print(f"saved a new best mape model at epoch {epoch}")

def get_outliers_idx(data, m=2):
    idxs = abs(data - np.mean(data)) > m * np.std(data)
    return idxs

def add_capacity(bdf, bdf_cap):
    bdf_cap = bdf_cap[(bdf_cap["type_of_capacity_calc"] == 1) & (bdf_cap["current_capacity"] > 0) & (bdf_cap["current_capacity"] > 0)]
    bdf["cap"] = np.interp(bdf["idx"].to_numpy(), bdf_cap["end_idx"], bdf_cap["current_capacity"])
    return bdf

def get_cell_bdf(cellname, data_dir):
    pklfile = os.path.join(data_dir, f"{cellname}_normal.pkl")
    bdf = pd.read_pickle(pklfile)
    
    csvfile = os.path.join(data_dir, f"{cellname}_normal_cap.csv")
    bdf_cap = pd.read_csv(csvfile)
    bdf = add_capacity(bdf, bdf_cap)

    return bdf

def get_all_bdfs(all_cellnames, data_dir, all_cells=False):
    all_bdfs_file = os.path.join(data_dir, "all_bdfs.pkl")
    if all_cells and os.path.isfile(all_bdfs_file):
        with open(all_bdfs_file,'rb') as fp:
            return pickle.load(fp)

    all_bdfs = {}
    for cellname in tqdm(all_cellnames):  
        all_bdfs[cellname] = get_cell_bdf(cellname, data_dir)
    
    if all_cells and not os.path.isfile(all_bdfs_file):
        with open(all_bdfs_file,'wb') as fp:
            pickle.dump(all_bdfs, fp)
    
    return all_bdfs

def get_wins_beg_idxs(bdf, window_length, window_step):
    return np.arange(window_length-1, len(bdf)-window_length, window_step)

def extract_dataset(all_bdfs, cellnames, window_length, window_step):
    """
        X.shape: n*3*window_length
        Y.shape: n
    """
    X = []
    Y = []

    for cellname in (cellnames):
        # print(f"------------------------------------ feature extraction for {cellname} ------------------------------------")
        bdf = all_bdfs[cellname]
        
        beg_idxs = get_wins_beg_idxs(bdf, window_length, window_step)
        end_idxs = beg_idxs + window_length - 1
        
        wins_dcaps = bdf.loc[end_idxs]["cap"].to_numpy() - bdf.loc[beg_idxs]["cap"].to_numpy()
        wins_dcaps = np.array(wins_dcaps*1e6)
        
        x = []
        for measurement_key in ["temperature", "voltage", "current"]:
            wins_x = np.array([bdf.loc[beg_idxs[k]:end_idxs[k]][measurement_key] for k in range(len(beg_idxs))]) # n*window_length
            wins_x = wins_x.reshape((-1, window_length, 1))
            # print("wins_x.shape: ", wins_x.shape)
            x.append(wins_x)
            
        x = np.concatenate(x, axis=2)
        # print("x.shape: ", x.shape)

        X.append(x)
        Y.append(wins_dcaps)
    
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y)
    # print("X.shape: ", X.shape)
    # print("Y.shape: ", Y.shape)
    return X, Y

def get_errors(pred, true):
    mae = np.abs(true - pred).mean()
    mape = np.abs((true - pred)/true).mean()*100

    return mae, mape

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default="cnn-1")
    parser.add_argument('--out-dir', type=str, default="exps/cnn-1")
    parser.add_argument('--data-dir', type=str, default="data")
    parser.add_argument('--train-cells', type=str, default=None)
    parser.add_argument('--valid-cells', type=str, default=None)
    parser.add_argument('--window-length', type=int, default=1024)
    parser.add_argument('--window-step', type=int, default=1024)
    parser.add_argument('--latent-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--nj', type=int, default=32)
    
    return parser 

def save_config_file(args):
    params_file = os.path.join(args.out_dir, "params_ae.json")
    with open(params_file, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

def read_train_test_cells(args):
    if args.train_cells is None or len(args.train_cells) == 0:
        with open(os.path.join(args.data_dir, "train_cells.txt")) as f:
            train_cellnames = [x.rstrip() for x in f.readlines()]
    else:
        train_cellnames = args.train_cells.split(",")
    if args.valid_cells is None or len(args.valid_cells) == 0:
        with open(os.path.join(args.data_dir, "test_cells.txt")) as f:
            valid_cellnames = [x.rstrip() for x in f.readlines()]
    else:
        valid_cellnames = args.valid_cells.split(",")

    return train_cellnames, valid_cellnames