#!/bin/bash

mkdir -p ./output_folder/logs


TIMESTAMP=$(date +%Y%m%d_%H%M%S)


TRAIN_LOG="./output_folder/logs/train_${TIMESTAMP}.log"
TEST_LOG="./output_folder/logs/test_${TIMESTAMP}.log"
TUNE_LOG="./output_folder/logs/tune_${TIMESTAMP}.log"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

model_names='mamba_mil'
backbones='resnet50'

declare -A in_dim
in_dim["resnet50"]=1024
in_dim["plip"]=512 
in_dim['ctp']=768

declare -A gpus
gpus["mean_mil"]=0
gpus["max_mil"]=0
gpus["att_mil"]=0
gpus["trans_mil"]=0
gpus['s4model']=0
gpus['mamba_mil']=0
gpus['mamba2_mil']=0

task="OSU"
model_size="small"
preloading="no"
patch_size="896"

loss_type='standard'
use_ensemble=false
ensemble_type='bagging'
num_models=5


lr_set='6e-5'

IFS=' ' read -r -a lr_array <<< "$lr_set"

mambamil_rate='5'
mambamil_layer='2'
mambamil_type='SRMamba'


PROJECT_ROOT="your_dir"


log_start_time() {
    local stage=$1
    local log_file=$2
    echo "===========================================" >> "$log_file"
    echo "Starting $stage stage at $(date)" >> "$log_file"
    echo "===========================================" >> "$log_file"
}


log_end_time() {
    local stage=$1
    local log_file=$2
    echo "===========================================" >> "$log_file"
    echo "Finished $stage stage at $(date)" >> "$log_file"
    echo "===========================================" >> "$log_file"
}


get_results_dir1() {
    local stage=$1
    local current_lr=$2
    local dir="./experiments/${stage}/${backbone}/${task}/${loss_type}/prototype/ant/${current_lr}/mlayer_${mambamil_layer}_mrate_${mambamil_rate}_0.5cla+0.5contrasive_skip_mlp_multiheadcross_t0.07_bag3"

    echo $dir
}


for model in $model_names; do
    for backbone in $backbones; do
        for current_lr in "${lr_array[@]}"; do
            exp=$model"/"$backbone
            echo "Training $exp with learning rate: $current_lr, GPU is: ${gpus[$model]}"
            export CUDA_VISIBLE_DEVICES=${gpus[$model]}
            
            train_results_dir=$(get_results_dir1 "train" "$current_lr")
            mkdir -p $train_results_dir
            
            log_start_time "Training" "$TRAIN_LOG"
            echo "Training $exp with learning rate: $current_lr" >> "$TRAIN_LOG"
            
            k_start=-1
            k_end=-1
            python main.py \
                --drop_out 0.1\
                --early_stopping \
                --lr $current_lr \
                --k 3 \
                --k_start $k_start \
                --k_end $k_end \
                --label_frac 1.0 \
                --exp_code $exp \
                --patch_size $patch_size \
                --weighted_sample \
                --task $task \
                --backbone $backbone \
                --results_dir $train_results_dir \
                --model_type $model \
                --log_data \
                --split_dir './splits/threefold_osu_over' \
                --preloading $preloading \
                --in_dim ${in_dim[$backbone]} \
                --mambamil_rate $mambamil_rate \
                --mambamil_layer $mambamil_layer \
                --mambamil_type $mambamil_type \
                --loss_type $loss_type \
                --seed 1 \
                --pseudo_threshold 0.9 >> "$TRAIN_LOG" 2>&1
            
            log_end_time "Training" "$TRAIN_LOG"


            echo "Testing $exp with learning rate: $current_lr, GPU is: ${gpus[$model]}"
            test_results_dir=$(get_results_dir1 "test" "$current_lr")
            mkdir -p $test_results_dir
            
            log_start_time "Testing" "$TEST_LOG"
            echo "Testing $exp with learning rate: $current_lr" >> "$TEST_LOG"
            
            python test.py \
                --task $task \
                --model_type $model \
                --backbone $backbone \
                --patch_size $patch_size \
                --exp_code $exp \
                --in_dim ${in_dim[$backbone]} \
                --results_dir $test_results_dir \
                --k 3 \
                --ckpt_path $train_results_dir \
                --split_dir './splits/threefold_osu_over' \
                --seed 1 \
                --drop_out 0.1 \
                --mambamil_rate $mambamil_rate \
                --mambamil_layer $mambamil_layer \
                --mambamil_type $mambamil_type \
                --loss_type $loss_type \
            
            log_end_time "Testing" "$TEST_LOG"

            tune_input_dir="${test_results_dir}/${model}"
            log_start_time "Tuning" "$TUNE_LOG"
            echo "Tuning $exp with learning rate: $current_lr" >> "$TUNE_LOG"
            
            python tune.py \
                --input_dir $tune_input_dir >> "$TUNE_LOG" 2>&1
            
            log_end_time "Tuning" "$TUNE_LOG"
        done
    done
done


SUMMARY_LOG="./output_folder/logs/summary_${TIMESTAMP}.log"
echo "Pipeline Execution Summary" > "$SUMMARY_LOG"
echo "Timestamp: $TIMESTAMP" >> "$SUMMARY_LOG"
echo "Learning rates used: ${lr_set}" >> "$SUMMARY_LOG"
echo "" >> "$SUMMARY_LOG"
echo "Log Files:" >> "$SUMMARY_LOG"
echo "Training Log: $TRAIN_LOG" >> "$SUMMARY_LOG"
echo "Testing Log: $TEST_LOG" >> "$SUMMARY_LOG"
echo "Tuning Log: $TUNE_LOG" >> "$SUMMARY_LOG"

echo "All stages completed. Log files:"
echo "Training: $TRAIN_LOG"
echo "Testing: $TEST_LOG"
echo "Tuning: $TUNE_LOG"
echo "Summary: $SUMMARY_LOG"