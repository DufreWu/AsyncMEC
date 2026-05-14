#!/bin/bash

function get_train_cellnames() { # args: ($data_dir)
    data_dir=$1;
    train_cells=$(cat $data_dir/train_cells.txt | tr "\n" "," | sed 's/,$//');
    echo $train_cells;
    return 0;
}

function get_valid_cellnames() { # args: ($data_dir)
    data_dir=$1;
    valid_cells=$(cat $data_dir/test_cells.txt | tr "\n" "," | sed 's/,$//');
    echo $valid_cells;
    return 0;
}

function get_all_cellnames() { # args: ($data_dir)
    data_dir=$1;
    train_cells=$(get_train_cellnames $data_dir);
    valid_cells=$(get_valid_cellnames $data_dir);
    echo "$train_cells,$valid_cells"
    return 0;
}


export TENSORBOARD_DIR="../exps/tb"

exp_name="ae-1"
train_cells=""
valid_cells=""
time_step=1
window_length=32768
window_step=32768
window_overlap=$(($window_length-$window_step))
latent_size=256
nj=8
data_dir="../data/tri"
epochs=100
lr=0.001
stage=-1
use_ae_dir=

if [ -f "$data_dir/train_cells.txt" ] && [ -f "$data_dir/test_cells.txt" ]; then
    train_cells=$(get_train_cellnames $data_dir)
    valid_cells=$(get_valid_cellnames $data_dir)
    all_cells=$(get_all_cellnames $data_dir)    
fi

echo "$0 $@"

. cmd.sh
. $NASA_SRC/parse_options.sh

exp_dir=../exps/$exp_name

mkdir -p $exp_dir

source ../../.venv/bin/activate
which python

if [ -n "${use_ae_dir}" ]; then
    ln -sr $use_ae_dir $exp_dir/ae
fi

ae_out_dir=$exp_dir/ae
dcap_out_dir=$exp_dir/dcap
mkdir -p $ae_out_dir $dcap_out_dir

if [ $stage -le -1 ]; then
    mkdir -p $data_dir
    python data_prep.py --tri-data-dir "$TRI_DATA_DIR" \
            --data-dir $data_dir --normalized-time-step $time_step --nj $nj \
            | tee $exp_dir/log.data_prep || exit 1;
fi

if [ $stage -le 0 ]; then
    python $NASA_SRC/train_ae.py --exp-name $exp_name \
                --out-dir $ae_out_dir \
                --data-dir $data_dir \
                --train-cells "$train_cells" \
                --valid-cells "$valid_cells" \
                --window-length $window_length \
                --latent-size $latent_size \
                --epochs $epochs \
                --lr $lr \
                --window-step $window_step --nj $nj \
                | tee $ae_out_dir/log.ae || exit 1
fi

if [ $stage -le 1 ]; then
    python $NASA_SRC/train_dcap.py --exp-name $exp_name \
                --out-dir $dcap_out_dir \
                --ae-dir $ae_out_dir \
                --data-dir $data_dir \
                --train-cells "$train_cells" \
                --valid-cells "$valid_cells" \
                --window-length $window_length --window-step $window_step \
                --latent-size $latent_size \
                --epochs $epochs \
                --lr $lr \
                --nj $nj | tee $dcap_out_dir/log.dcap || exit 1

    # cat $dcap_out_dir/best-mape-model.txt | sort -t "," -k 2 -n | head
fi

if [ $stage -le 2 ]; then
    python $NASA_SRC/infer_dcap.py --exp-name $exp_name \
                --out-dir $dcap_out_dir/preds \
                --dcap-dir $dcap_out_dir \
                --data-dir $data_dir \
                --train-cells "$train_cells" \
                --valid-cells "$valid_cells" \
                --window-length $window_length --window-step $window_length \
                --latent-size $latent_size \
                --nj $nj | tee $dcap_out_dir/log.dcap.pred || exit 1

fi