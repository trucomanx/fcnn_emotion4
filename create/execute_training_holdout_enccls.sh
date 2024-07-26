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

WFile=$BaseDir'/OUTPUTS/DOCTORADO2/fcnn_emotion4_1/ber2024-skel/training_validation_holdout/encdec_ncod20_1/model_encoder.h5'

################################################################################

ipynb-py-convert training_holdout_enccls.ipynb training_holdout_enccls.py

python3 training_holdout_enccls.py  --epochs  10 \
                                    --patience 2000 \
                                    --seed 0 \
                                    --batch-size 2048 \
                                    --subdir enccls_ncod20 \
                                    --ncod 20 \
                                    --weights-init $WFile \
                                    --dataset-train-dir $InTrD \
                                    --dataset-train-file $InTrF \
                                    --dataset-test-dir $InTsD \
                                    --dataset-test-file $InTsF \
                                    --dataset-name $DName \
                                    --output-dir $OutDir


rm -f training_holdout_enccls.py

