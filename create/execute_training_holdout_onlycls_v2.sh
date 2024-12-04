#!/bin/bash

PyCommand='#!/usr/bin/python3

json_filename="training_data_results.json"

model_list=["onlycls_ncod15",
            "onlycls_ncod18",
            "onlycls_ncod20",
            "onlycls_ncod22",
            "onlycls_ncod25",
            "onlycls_ncod30"
            ];

info_list=[ "train_categorical_accuracy",
            "train_loss",
            "val_categorical_accuracy",
            "val_loss",
            "test_categorical_accuracy",
            "test_loss",
            "number_of_parameters"
            ];

sep=",";

image_ext=".eps";
'

# HD
BaseDir='/media/fernando/Expansion'
# 
#BaseDir='/media/fernando/B0EA304AEA300EDA/Dados/Fernando'

OutDir=$BaseDir'/OUTPUTS/DOCTORADO2/fcnn_emotion4_v2'

DName='ber2024-skel'

InTrD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-SKELETON'
InTrF='train_refface.csv'
InTsD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-SKELETON'
InTsF='test_refface.csv'

################################################################################

mkdir -p $OutDir/$DName/training_validation_holdout
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/training_validation_holdout/'main.py'

################################################################################

ipynb-py-convert training_holdout_onlycls.ipynb training_holdout_onlycls.py

for ncod in 11 15 18 20 22 25 29; do
    echo " "
    python3 training_holdout_onlycls.py --epochs  10000 \
                                        --patience 2000 \
                                        --seed 0 \
                                        --ncod $ncod \
                                        --batch-size 2048 \
                                        --dataset-train-dir $InTrD \
                                        --dataset-train-file $InTrF \
                                        --dataset-test-dir $InTsD \
                                        --dataset-test-file $InTsF \
                                        --dataset-name $DName \
                                        --output-dir $OutDir
done




rm -f training_holdout_onlycls.py

