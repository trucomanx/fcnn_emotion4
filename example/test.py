#!/usr/bin/python3

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import sys
sys.path.append('../library');

import SkeletonEmotion4Lib.Classifier as sec
import numpy as np

cls=sec.Emotion4Classifier(ncod=20);

vec=np.random.randn(51);

print("")
print("cls.predict_vec(vec)")
res=cls.predict_vec(vec);
print(res);


print("")
print("cls.from_skel_npvector_list([vec,vec])")
res=cls.from_skel_npvector_list([vec,vec]);
print(res);

print("")
print("cls.predict_minus_vec(vec)")
res=cls.predict_minus_vec(vec);
print(res);

print("")
print("cls.predict_vec_list([vec,vec])")
res=cls.predict_vec_list([vec,vec]);
print(res);

print("")
print("cls.predict_minus_vec_list([vec,vec])")
res=cls.predict_minus_vec_list([vec,vec]);
print(res);

