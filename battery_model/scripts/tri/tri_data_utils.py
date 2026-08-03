import h5py
import numpy as np
import pickle
import pandas as pd 
from tqdm import tqdm
import os
import sys

tri_data_batchid2file = {
    "1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat", 
    "2": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
    "4": "2019-01-24_batchdata_updated_struct_errorcorrect.mat"
}

tri_noisy_cells = {
    "1": ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"], 
    "2": ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16'],
    "3": ["b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43"],
    "4": ["b4c31"]
}

def load_tri_data_mat2df(batch_id, matfile, normalized_time_step=0.1):
    """
        loads a matlab data file (for example: 2017-05-12_batchdata_updated_struct_errorcorrect.mat) containing ~48 cells' data
        into a dict by cells' id as key and dataframes as values.
    """
    print(matfile)            
    with h5py.File(matfile) as f:
        print(f.keys())
        batch = f['batch']
        print(batch.keys())
        
        num_cells = batch['summary'].shape[0]
        bat_dict = {}
        for i in tqdm(range(num_cells)):
            key = f'b{batch_id}c' + str(i)
            if key in tri_noisy_cells[batch_id]:
                continue

            cl = f[batch['cycle_life'][i,0]][()][0][0]
            policy = f[batch['policy_readable'][i,0]][()].tobytes()[::2].decode()
            summary_IR = np.hstack(f[batch['summary'][i,0]]['IR'][0,:].tolist())
            summary_QC = np.hstack(f[batch['summary'][i,0]]['QCharge'][0,:].tolist())
            summary_QD = np.hstack(f[batch['summary'][i,0]]['QDischarge'][0,:].tolist())
            summary_TA = np.hstack(f[batch['summary'][i,0]]['Tavg'][0,:].tolist())
            summary_TM = np.hstack(f[batch['summary'][i,0]]['Tmin'][0,:].tolist())
            summary_TX = np.hstack(f[batch['summary'][i,0]]['Tmax'][0,:].tolist())
            summary_CT = np.hstack(f[batch['summary'][i,0]]['chargetime'][0,:].tolist())
            summary_CY = np.hstack(f[batch['summary'][i,0]]['cycle'][0,:].tolist())
            summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg':
                        summary_TA, 'Tmin': summary_TM, 'Tmax': summary_TX, 'chargetime': summary_CT,
                        'cycle': summary_CY}
            cycles = f[batch['cycles'][i,0]]
            
            cycles_data = []
            cycles_data_normal = []
            init_time = 0.0
            for j in range(cycles['I'].shape[0]):
                I = np.hstack((f[cycles['I'][j,0]][()]))
                T = np.hstack((f[cycles['T'][j,0]][()]))
                V = np.hstack((f[cycles['V'][j,0]][()]))
                relt = np.hstack((f[cycles['t'][j,0]][()]))*60
                times = relt + init_time
                cycle_number = j+1
                init_time = times[-1]

                d = list(zip(times, relt, V, I, T, [cycle_number]*len(relt)))
                cycles_data.extend(d)
                
                if normalized_time_step is not None:
                    normalized_time_step = np.float64(normalized_time_step)
                    t0 = relt[0]
                    tn = relt[-1]
                    tn = tn - tn % normalized_time_step
                    n = int((tn-t0) / normalized_time_step) + 1
                    normal_reltimes = np.linspace(t0, tn, n)
                    y_normal = list(zip(
                        np.interp(normal_reltimes, relt, times),
                        normal_reltimes,
                        np.interp(normal_reltimes, relt, V), 
                        np.interp(normal_reltimes, relt, I), 
                        np.interp(normal_reltimes, relt, T), 
                        [cycle_number]*len(normal_reltimes)
                    ))
                    cycles_data_normal.extend(y_normal)
        
            df = pd.DataFrame(cycles_data, columns=["time", "relativeTime", "voltage", "current", "temperature", "cycle_number"])
            df = df[(df[["time","relativeTime","voltage","current","temperature"]] != 0).any(axis=1)]
            df['idx'] = df.index

            df_normal = pd.DataFrame(cycles_data_normal, columns=["time", "relativeTime", "voltage", "current", "temperature", "cycle_number"])
            df_normal = df_normal[(df_normal[["time","relativeTime","voltage","current","temperature"]] != 0).any(axis=1)]
            df_normal['idx'] = df_normal.index            

            cell_dict = {'cycle_life': cl, 'charge_policy':policy, 'summary': summary, 'cycles_data': df, "cycles_data_normal": df_normal}
            
            bat_dict[key] = cell_dict
            
    return bat_dict


def create_capacity_file(bdf_normal, summary, csvfile_cap, filter_peaks=False):
    cycles = summary["cycle"]
    caps = summary["QD"]
    cap_moving_avg = caps[caps>0][0]
    skipped = 0
    with open(csvfile_cap, mode="w") as f:
        print("beg_idx,end_idx,beg_time,end_time,current_capacity,type_of_capacity_calc", file=f)
        for i, cycle in enumerate(cycles):
            curr_capacity = caps[i]
            if curr_capacity == 0:
                continue
            if filter_peaks and abs(curr_capacity - cap_moving_avg) > 0.1*cap_moving_avg:
                skipped += 1
                continue
            cap_moving_avg = 0.7*cap_moving_avg + 0.3*curr_capacity

            beg_idx = bdf_normal[bdf_normal["cycle_number"] == cycle]["idx"].min()
            end_idx = bdf_normal[bdf_normal["cycle_number"] == cycle]["idx"].max()

            beg_time = bdf_normal["time"].loc[beg_idx]                    
            end_time = bdf_normal["time"].loc[end_idx]
            
            print(f"{beg_idx},{end_idx},{beg_time},{end_time},{curr_capacity},1", file=f) # RD: reference discharge
    print(f"\ncreate_capacity_file: skipped {skipped} (from {len(caps)}) in {os.path.basename(csvfile_cap)} capacity measurements due to high deviation from moving average", file=sys.stdout, flush=True)
        
def create_grouped_cycle_number_file(bdf_normal, summary, csvfile_gc):
    # let's assume the experiment is one big cycle because actual cycles are too short (60 secs) that does not incorporate as one window with size
    cycles = summary["cycle"]
    caps = summary["QD"]
    with open(csvfile_gc, mode="w") as f:
        print("beg_idx,end_idx,grouped_cycle_number,grouped_cycle_type", file=f)
        beg_idx = bdf_normal["idx"].min()
        end_idx = bdf_normal["idx"].max()
        print(f"{beg_idx},{end_idx},{1},-", file=f)
    
def create_windowing_start_indices(gcfile, window_length, window_overlap, outfile):
    gc_bdf = pd.read_csv(gcfile)
    with open(outfile, mode="w") as f:
        print("idx", file=f)
        
        for i,row in gc_bdf.iterrows():
            beg_idx = row["beg_idx"]
            end_idx = row["end_idx"]
            clen = end_idx - beg_idx + 1
            if clen >= window_length:
                end_wsidx = (clen-window_length) - (clen%window_overlap) 
            else:
                end_wsidx = -1
            wsidxs = beg_idx + np.arange(0, end_wsidx+1, (window_length-window_overlap))
            for wsidx in wsidxs:
                print(wsidx, file=f)
    return outfile

