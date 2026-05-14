#!/usr/bin/python3

import os
import sys
import argparse
import glob
from multiprocessing import Pool
import re

from tri_data_utils import *

def prepare_batch_mat_file(args, batch_id, matfilename):
    matfile = os.path.join(args.tri_data_dir, matfilename)
    print(matfile)

    dcells = load_tri_data_mat2df(batch_id, matfile, normalized_time_step=args.normalized_time_step)

    for cell_key, cell_data in dcells.items():
        csvfile = os.path.join(args.data_dir, f"{cell_key}.csv")
        bdf = cell_data["cycles_data"]
        bdf.to_csv(csvfile, index=False)
        
        bdf_normal = cell_data["cycles_data_normal"]
        csvfile_normal = os.path.join(args.data_dir, f"{cell_key}_normal.csv")
        pklfile_normal = os.path.join(args.data_dir, f"{cell_key}_normal.pkl")
    
        bdf_normal.to_csv(csvfile_normal, index=False)
        bdf_normal.to_pickle(pklfile_normal)

        csvfile_cap = os.path.join(args.data_dir, f"{cell_key}_normal_cap.csv")
        create_capacity_file(bdf_normal, cell_data["summary"], csvfile_cap, filter_peaks=True)

        csvfile_gc = os.path.join(args.data_dir, f"{cell_key}_normal_gc.csv")
        create_grouped_cycle_number_file(bdf_normal, cell_data["summary"], csvfile_gc)
        
        # if args.window_length is not None:
        #     csvfile_wsidx = os.path.join(args.data_dir, f"{cell_key}_normal_wsidx.csv")
        #     create_windowing_start_indices(csvfile_gc, args.window_length, args.window_overlap, csvfile_wsidx)
        
        print(cell_key, " done.", file=sys.stdout, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tri-data-dir', type=str)
    parser.add_argument('--data-dir', type=str, default="data")
    parser.add_argument('--normalized-time-step', type=float, default=None)
    parser.add_argument('--nj', type=int, default=1, help="number of jobs")
    
    args = parser.parse_args()
    print(" ".join(sys.argv))
    with open(os.path.join(args.data_dir, "normalized_timestep"), mode="w") as f:
        print(args.normalized_time_step, file=f)
    
    
    # for batch_id, matfilename in tri_data_batchid2file.items():
    #     prepare_batch_mat_file(args, batch_id, matfilename)

    with Pool(args.nj) as p:
        print(p.starmap(
                prepare_batch_mat_file, 
                zip([args]*len(tri_data_batchid2file), [bid for bid in tri_data_batchid2file], [matfilename for _,matfilename in tri_data_batchid2file.items()])
            )
        )

    def get_batch_batnames(batch_id, all_batnames):
        return [batname for batname in all_batnames if re.search(rf'b{batch_id}c[0-9]+', batname)]

    all_csv_files = [f for f in glob.glob(os.path.join(args.data_dir, "*.pkl")) if re.search(r'b[0-9]+c[0-9]+_normal\.pkl', f)]
    all_batnames = [re.search(r'b[0-9]+c[0-9]+', os.path.basename(f))[0] for f in all_csv_files]

    traincells_file = os.path.join(args.data_dir, "train_cells.txt")
    with open(traincells_file, mode="w") as f:
        for batch_id in ["1", "2", "3"]:
            for cell_k in get_batch_batnames(batch_id, all_batnames):
                print(cell_k, file=f)

    testcells_file = os.path.join(args.data_dir, "test_cells.txt")
    with open(testcells_file, mode="w") as f:
        for batch_id in ["4"]:
            for cell_k in get_batch_batnames(batch_id, all_batnames):
                print(cell_k, file=f)
    

    print("all done.")