#!/usr/bin/python3

import os
import sys
import argparse
import glob
from multiprocessing import Pool
import re

from nasa_data_utils import *

def prepare_mat_file(args, matfile):
    print(matfile)
    batname = re.search(r'RW[0-9]+', os.path.basename(matfile))[0]
    csvfile = os.path.join(args.data_dir, f"{batname}.csv")
    csvfile_normal = os.path.join(args.data_dir, f"{batname}_normal.csv")
    pklfile_normal = os.path.join(args.data_dir, f"{batname}_normal.pkl")
    bdf, bdf_normal = mat2df(matfile, normalized_delta_time=args.normalized_time_step)

    bdf.to_csv(csvfile, index=False)
    bdf_normal.to_csv(csvfile_normal, index=False)
    bdf_normal.to_pickle(pklfile_normal)

    csvfile_cap = os.path.join(args.data_dir, f"{batname}_normal_cap.csv")
    create_capacity_file(bdf_normal, csvfile_cap)

    csvfile_gc = os.path.join(args.data_dir, f"{batname}_normal_gc.csv")
    create_grouped_cycle_number_file(bdf_normal, csvfile_gc)
    
    # if args.window_length is not None:
    #     csvfile_wsidx = os.path.join(args.data_dir, f"{batname}_normal_wsidx.csv")
    #     create_windowing_start_indices(csvfile_gc, args.window_length, args.window_overlap, csvfile_wsidx)
    print(matfile, " done.")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nasa-data-dir', type=str)
    parser.add_argument('--data-dir', type=str, default="data")
    parser.add_argument('--normalized-time-step', type=float, default=None)
    parser.add_argument('--window-length', type=int, default=None, help="number of samples in a processing window (None doesn't create the wsidx files)")
    parser.add_argument('--window-overlap', type=int, default=None, help="number of samples that two consecutive windows share")
    parser.add_argument('--nj', type=int, default=1, help="number of jobs")
    
    args = parser.parse_args()
    print(" ".join(sys.argv))
    with open(os.path.join(args.data_dir, "normalized_timestep"), mode="w") as f:
        print(args.normalized_time_step, file=f)
    
    all_mat_files = glob.glob(os.path.join(args.nasa_data_dir, "*", "data", "Matlab", "*.mat"))
    with Pool(args.nj) as p:
        print(p.starmap(prepare_mat_file, zip([args]*len(all_mat_files), all_mat_files)))

    traincells_file = os.path.join(args.data_dir, "train_cells.txt")
    with open(traincells_file, mode="w") as f:
        for cell_k in def_train_cells.split(","):
            print(cell_k, file=f)

    testcells_file = os.path.join(args.data_dir, "test_cells.txt")
    with open(testcells_file, mode="w") as f:
        for cell_k in def_valid_cells.split(","):
            print(cell_k, file=f)

    print("all done.")