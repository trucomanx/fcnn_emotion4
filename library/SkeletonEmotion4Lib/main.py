#!/usr/bin/python

import os
os.environ["TF_USE_LEGACY_KERAS"]="1";

import lib_model as mpp

################################################################################

model_encdec=mpp.create_model_encdec(load_weights=False,file_of_weight='',ncod=15)

print("endec",type(model_encdec))
model_encdec.summary();
model_encdec.save_weights("model_encdec.h5")

print("encoder",type(model_encdec.layers[0]))
model_encdec.layers[0].summary();
model_encdec.layers[0].save_weights("model_encoder.h5")


print("decoder",type(model_encdec.layers[1]))
model_encdec.layers[1].summary();
model_encdec.layers[1].save_weights("model_decoder.h5")

################################################################################

model_encoder=mpp.create_model_encoder(load_weights=False,file_of_weight="model_encoder.h5",ncod=15);
model_decoder=mpp.create_model_decoder(load_weights=False,file_of_weight="model_decoder.h5",ncod=15);
