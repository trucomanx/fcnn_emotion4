#!/usr/bin/python3

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import sys
sys.path.append('../library');

import SkeletonEmotion4Lib.Classifier as sec
import numpy as np

cls=sec.Emotion4Classifier();

vec=np.random.randn(51);

res=cls.predict_vec(vec);

print(res);

res=cls.predict_minus_vec(vec);

print(res);


