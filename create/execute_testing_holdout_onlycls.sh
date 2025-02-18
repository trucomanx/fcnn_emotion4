#!/bin/bash

PyCommand='#!/usr/bin/python3

json_filename="testing_data_results.json"

model_list=["onlycls_ncod15",
            "onlycls_ncod18",
            "onlycls_ncod20",
            "onlycls_ncod22",
            "onlycls_ncod25"
            ];

info_list=[ "block_delayms",
            "categorical_accuracy",
            "loss",
            "number_of_parameters"
            ];

sep=",";

image_ext=".eps";
'

# HD
#BaseDir='/media/fernando/Expansion'
BaseDir='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando'
#BaseDir='/media/fernando/B0EA304AEA300EDA/Dados/Fernando'

OutDir=$BaseDir'/OUTPUTS/DOCTORADO2/fcnn_emotion4_1_10times'

DName='ber2024-skel'


InTsD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-SKELETON'
InTsF='test.csv'
ModD=$BaseDir'/OUTPUTS/DOCTORADO2/SKEL/fcnn_emotion4/ber2024-skel/training_validation_holdout'

################################################################################

mkdir -p $OutDir/$DName/testing_holdout
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/testing_holdout/'main.py'

################################################################################

ipynb-py-convert testing_holdout_onlycls.ipynb testing_holdout_onlycls.py

for ncod in 15 18 20 22 25; do
    echo " "
    python3 testing_holdout_onlycls.py  --ncod $ncod \
                                        --model-file $ModD/'onlycls_ncod'$ncod/'model_onlycls_ncod'$ncod'.h5' \
                                        --times 10 \
                                        --batch-size 16 \
                                        --dataset-test-dir $InTsD \
                                        --dataset-test-file $InTsF \
                                        --dataset-name $DName \
                                        --output-dir $OutDir
done




rm -f testing_holdout_onlycls.py

