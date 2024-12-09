#!/bin/bash

PyCommand='#!/usr/bin/python3

json_filename="kfold_data_results.json"

model_list=["onlycls_ncod15",
            "onlycls_ncod18",
            "onlycls_ncod20",
            "onlycls_ncod22",
            "onlycls_ncod25",
            "onlycls_ncod29",
            "onlycls_ncod33",
            "onlycls_ncod37",
            "onlycls_ncod41",
            "onlycls_ncod45"
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
BaseDir='/media/maquina02/HD/Dados/Fernando'
# 
#BaseDir='/media/fernando/B0EA304AEA300EDA/Dados/Fernando'

OutDir=$BaseDir'/OUTPUTS/DOCTORADO2/fcnn_emotion4_v2'


DName='ber2024-skel'


if [ "$DName" = "ber2024-skel" ]; then
    InTrD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-SKELETON'
    InTrF='train_refface.csv'
fi

################################################################################

mkdir -p $OutDir/$DName/cross-validation
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/cross-validation/'main.py'

################################################################################

ipynb-py-convert kfold_validation_onlycls.ipynb kfold_validation_onlycls.py

# 11 15 18 20 22 25 29 33 37 41 45 49 53
for ncod in 11 15 18 20 22 25 29 33 37 41 45 49 53; do
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

