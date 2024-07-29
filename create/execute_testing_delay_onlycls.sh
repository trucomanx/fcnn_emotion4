#!/bin/bash

PyCommand='#!/usr/bin/python3

json_filename="testing_data_results.json"

model_list=["onlycls_ncod15",
            "onlycls_ncod18",
            "onlycls_ncod20",
            "onlycls_ncod22",
            "onlycls_ncod25"
            ];

info_list=[ "delayms",
            "categorical_accuracy"];

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
    InTsD=$BaseDir'/DATASET/TESE/BER/BER2024/BER2024-SKELETON'
    InTsF='test.csv'
    ModD=$BaseDir'/OUTPUTS/DOCTORADO2/fcnn_emotion4/ber2024-skel/training_validation_holdout'
fi

################################################################################

mkdir -p $OutDir/$DName/test_delay
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/test_delay/'main.py'

################################################################################

ipynb-py-convert testing_delay_onlycls.ipynb testing_delay_onlycls.py

for ncod in 15 18 20 22 25; do
    echo " "
    python3 testing_delay_onlycls.py    --model-file $ModD/'onlycls_ncod'$ncod/'model_onlycls_ncod'$ncod'.h5' \
                                        --times 10 \
                                        --ncod $ncod \
                                        --dataset-dir $InTsD \
                                        --dataset-file $InTsF \
                                        --dataset-name $DName \
                                        --output-dir $OutDir
done

rm -f testing_delay_onlycls.py

