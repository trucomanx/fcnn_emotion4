#!/bin/bash

# HD
BaseDir='/media/fernando/Expansion'
# 
#BaseDir='/media/fernando/B0EA304AEA300EDA/Dados/Fernando'

OutDir=$BaseDir'/OUTPUTS/DOCTORADO2/fcnn_emotion4_1'

DName='ber2024-skel'

InTrD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-SKELETON'
InTrF='train.csv'
InTsD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-SKELETON'
InTsF='test.csv'


################################################################################

ipynb-py-convert training_holdout.ipynb training_holdout.py

python3 training_holdout.py --epochs 40 \ 
                            --batch-size 1024 \ 
                            --dataset-train-dir $InTrD \ 
                            --dataset-train-file $InTrF \ 
                            --dataset-test-dir $InTsD \ 
                            --dataset-test-file $InTsF \ 
                            --dataset-name $DName \ 
                            --output-dir $OutDir

#rm -f training_holdout.py

