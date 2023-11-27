


# In be noises out be svhn
cat ~/CSI/finals/noise_evaluation/available_noises.txt | while read -r noise
do 
    python binary_comparision.py --in_dataset $noise --shift $noise --ood 'svhn'  --mode 'binary' --model_path '/storage/users/makhavan/CSI/exp04/scone_binary/CIFAR/run/auc_98/best_params.pt' > tmp_binary.log
    auc=$(cat tmp_binary.log | grep "Evaluation/avg_auc:" | cut -d ':' -f2)
    echo -e "$noise: $auc\n" >> shifts_in_svhn_out_binary.log
done

cat ~/CSI/finals/noise_evaluation/available_noises.txt | while read -r noise
do 
    python binary_comparision.py --in_dataset $noise --shift $noise --ood 'svhn'  --mode 'classification' --model_path '/storage/users/makhavan/CSI/finals/scone_original_repo/scone/CIFAR/checkpoints/cifar10/svhn/scone/scone_1_1_0.05_1_1_1.5_0.5_0.1_epoch_49.pt' > tmp_scone.log
    auc_scone=$(cat tmp_scone.log | grep "Evaluation/avg_auc:" | cut -d ':' -f2)
    echo -e "$noise: $auc_scone\n" >> shifts_in_svhn_out_scone.log
done

# In be cifar10 out be svhn
python binary_comparision.py --in_dataset 'cifar10' --ood 'svhn'  --mode 'binary' --model_path '/storage/users/makhavan/CSI/exp04/scone_binary/CIFAR/run/auc_98/best_params.pt' > tmp_binary.log
auc=$(cat tmp_binary.log | grep "Evaluation/avg_auc:" | cut -d ':' -f2)
echo -e "cifar10: $auc\n" >> cifar10_in_svhn_out_binary.log

python binary_comparision.py --in_dataset 'cifar10' --ood 'svhn'  --mode 'classification' --model_path '/storage/users/makhavan/CSI/finals/scone_original_repo/scone/CIFAR/checkpoints/cifar10/svhn/scone/scone_1_1_0.05_1_1_1.5_0.5_0.1_epoch_49.pt' > tmp_scone.log
auc=$(cat tmp_scone.log | grep "Evaluation/avg_auc:" | cut -d ':' -f2)
echo -e "cifar10: $auc\n" >> cifar10_in_svhn_out_scone.log

# In be cifar10 out be shifts
cat ~/CSI/finals/noise_evaluation/available_noises.txt | while read -r noise
do 
    python binary_comparision.py --in_dataset 'cifar10' --shift $noise --ood $noise  --mode 'binary' --model_path '/storage/users/makhavan/CSI/exp04/scone_binary/CIFAR/run/auc_98/best_params.pt' > tmp_binary.log
    auc=$(cat tmp_binary.log | grep "Evaluation/avg_auc:" | cut -d ':' -f2)
    echo -e "$noise: $auc\n" >> cifar10_in_shifts_out_binary.log
done

cat ~/CSI/finals/noise_evaluation/available_noises.txt | while read -r noise
do 
    python binary_comparision.py --in_dataset 'cifar10' --shift $noise --ood $noise  --mode 'classification' --model_path '/storage/users/makhavan/CSI/finals/scone_original_repo/scone/CIFAR/checkpoints/cifar10/svhn/scone/scone_1_1_0.05_1_1_1.5_0.5_0.1_epoch_49.pt' > tmp_scone.log
    auc_scone=$(cat tmp_scone.log | grep "Evaluation/avg_auc:" | cut -d ':' -f2)
    echo -e "$noise: $auc_scone\n" >> cifar10_in_shifts_out_scone.log
done
