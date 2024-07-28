#!/bin/bash

PyCommand='#!/usr/bin/python3

json_filename="kfold_data_results.json"

model_list=["onlycls_ncod18",
            "onlycls_ncod20",
            "onlycls_ncod22"
            ];

info_list=[ "mean_val_categorical_accuracy",
            "std_val_categorical_accuracy",
            "mean_val_loss",
            "mean_train_categorical_accuracy",
            "mean_train_loss",
            "number_of_parameters"];

erro_bar=[("mean_val_categorical_accuracy","std_val_categorical_accuracy")];

p_matrix="val_categorical_accuracy";

sep=",";

image_ext=".eps";
'

# HD
BaseDir='/media/fernando/Expansion'
# 
#BaseDir='/media/fernando/B0EA304AEA300EDA/Dados/Fernando'

OutDir=$BaseDir'/OUTPUTS/DOCTORADO2/fcnn_emotion4_1'


DName='ber2024-skel'


if [ "$DName" = "ber2024-skel" ]; then
    InTrD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-SKELETON'
    InTrF='train.csv'
fi

################################################################################

mkdir -p $OutDir/$DName/cross-validation
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/cross-validation/'main.py'

################################################################################

ipynb-py-convert kfold_validation_onlycls.ipynb kfold_validation_onlycls.py

# 15 18 20 22
for ncod in 25 11 29; do
    echo " "
    python3 kfold_validation_onlycls.py --epochs  10000 \
                                        --patience 2000 \
                                        --seed 0 \
                                        --ncod $ncod \
                                        --batch-size 2048 \
                                        --dataset-dir $InTrD \
                                        --dataset-file $InTrF \
                                        --dataset-name $DName \
                                        --output-dir $OutDir

done

rm -f kfold_validation_onlycls.py

