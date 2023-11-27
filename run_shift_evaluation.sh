#!/bin/bash

cat ~/CSI/finals/noise_evaluation/available_noises.txt | while read -r noise
do 
    echo "################## $noise ##################"
    python noise_evaluation.py --shift $noise --model_path 'run/exp_2023_11_26_12_51_01_728339__0)___lr_0.1_lrur_5.0_lrg_0.5_sgd/best_params.pt'  > tmp_noise_evaluation.log
    # auc=$(cat tmp_noise_evaluation.log | grep "Evaluation/avg_auc:" | cut -d ':' -f2)
    # echo -e "$noise: $auc\n" >> noise_evaluation_report.log
done
