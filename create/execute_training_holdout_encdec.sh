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

WFile=$BaseDir'/OUTPUTS/DOCTORADO2/fcnn_emotion4_1/ber2024-skel/training_validation_holdout_encdec/encdec_ncod20_1/model_encdec.h5'


################################################################################

ipynb-py-convert training_holdout_encdec.ipynb training_holdout_encdec.py

python3 training_holdout_encdec.py  --epochs  10000 \
                                    --patience 1000 \
                                    --seed 0 \
                                    --batch-size 2048 \
                                    --subdir encdec_ncod20_2 \
                                    --ncod 20 \
                                    --weights-init $WFile \
                                    --dataset-train-dir $InTrD \
                                    --dataset-train-file $InTrF \
                                    --dataset-test-dir $InTsD \
                                    --dataset-test-file $InTsF \
                                    --dataset-name $DName \
                                    --output-dir $OutDir


rm -f training_holdout_encdec.py

