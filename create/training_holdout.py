# %%
import os
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

os.environ['TF_USE_LEGACY_KERAS'] = '1';

# %%
input_default_json_conf_file='fcnn_emotion4_training_default.json';

# %%
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import json

# %%
import sys
sys.path.append('../library');

# %%
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %%
## Load json conf json file
fd = open(os.path.join('./',input_default_json_conf_file));
DATA = json.load(fd);
fd.close()

# %%
"""
# Variable globales
"""

# %%
## Seed for the random variables
seed_number=0;

## Dataset 
dataset_base_dir    = DATA['dataset_train_base_dir'];
dataset_labels_file = DATA['dataset_train_labels_file'];

dataset_base_test_dir    = DATA['dataset_test_base_dir'];
dataset_labels_test_file = DATA['dataset_test_labels_file'];

dataset_name        = DATA['dataset_name'];

## Training hyperparameters
EPOCAS     = DATA["epochs"];
BATCH_SIZE = DATA["batch_size"];


## Output
output_base_dir = DATA["output_base_dir"];


NCOD=15;
ANGLE=60;

# %%
"""
# Parametros de entrada
"""

# %%
for n in range(len(sys.argv)):
    if sys.argv[n]=='--dataset-train-dir':
        dataset_base_dir=sys.argv[n+1];
    elif sys.argv[n]=='--dataset-train-file':
        dataset_labels_file=sys.argv[n+1];
    elif sys.argv[n]=='--dataset-test-dir':
        dataset_base_test_dir=sys.argv[n+1];
    elif sys.argv[n]=='--dataset-test-file':
        dataset_labels_test_file=sys.argv[n+1];
    elif sys.argv[n]=='--dataset-name':
        dataset_name=sys.argv[n+1];
    elif sys.argv[n]=='--epochs':
        EPOCAS=int(sys.argv[n+1]);
    elif sys.argv[n]=='--batch-size':
        BATCH_SIZE=int(sys.argv[n+1]);
    elif sys.argv[n]=='--output-dir':
        output_base_dir=sys.argv[n+1];
        
print('        dataset_base_dir:',dataset_base_dir)
print('     dataset_labels_file:',dataset_labels_file)
print('   dataset_base_test_dir:',dataset_base_test_dir)
print('dataset_labels_test_file:',dataset_labels_test_file)
print('            dataset_name:',dataset_name)
print('                  EPOCAS:',EPOCAS)
print('              BATCH_SIZE:',BATCH_SIZE)
print('         output_base_dir:',output_base_dir)

# %%
"""
# Set seed of random variables

"""

# %%
np.random.seed(seed_number)
tf.keras.utils.set_random_seed(seed_number);

# %%
"""
# Loading data of dataset
"""

# %%
# Load filenames and labels
train_val_df = pd.read_csv(os.path.join(dataset_base_dir,dataset_labels_file));
#print(train_val_df)

# Setting labels
Y = train_val_df[['label']];
L=np.shape(Y)[0];

# Load test filenames and labels
test_df = pd.read_csv(os.path.join(dataset_base_test_dir,dataset_labels_test_file));

print('\n\ntest_df')
print(test_df)


# %%
"""
# Setting the cross-validation split

"""

# %%
from sklearn.model_selection import train_test_split

training_df, validation_df = train_test_split(train_val_df, test_size=0.2,shuffle=True, stratify=Y)

print('\n\ntraining_df')
print(training_df);


print('\n\nvalidation_df')
print(validation_df)

# %%
"""
# Data augmentation configuration
"""

# %%
import SkeletonEmotion4Lib.DataAugmentation as sda
import SkeletonEmotion4Lib.lib_tools as slt

training_data_array   = slt.batch_normalize_coordinates(training_df.iloc[:,0:51].values)
validation_data_array = slt.batch_normalize_coordinates(validation_df.iloc[:,0:51].values)
test_data_array       = slt.batch_normalize_coordinates(test_df.iloc[:,0:51].values)

print('train:',training_data_array.shape)
print('val:',validation_data_array.shape)
print('test:',test_data_array.shape)

train_data_generator = sda.DataAugmentationEncDecGenerator(training_data_array, 
                                    batch_size=BATCH_SIZE, 
                                    augment_fn=lambda X: slt.batch_random_rotate_coordinates(X,angle=ANGLE));

valid_data_generator = sda.DataAugmentationEncDecGenerator(validation_data_array, 
                                        batch_size=BATCH_SIZE, 
                                        augment_fn=None);

test_data_generator = sda.DataAugmentationEncDecGenerator(   test_data_array, 
                                            batch_size=BATCH_SIZE, 
                                            augment_fn=None);




# %%
"""
# Creating output directory
"""

# %%
output_dir = os.path.join(output_base_dir,dataset_name,'training_validation_holdout');

os.makedirs(output_base_dir,exist_ok = True);

os.makedirs(output_dir,exist_ok = True);

# %%
"""
# Create new encoder-decoder model
"""

# %%
import SkeletonEmotion4Lib.lib_model as mpp

model = mpp.create_model_encdec(load_weights=False,file_of_weight='',ncod=NCOD);

print('input_shape',model.input_shape)
print('output_shape',model.output_shape)
print('')

model.summary()

mpp.save_model_parameters(model, os.path.join(output_dir,'parameters_stats_endec.m'));


# %%
# COMPILE NEW MODEL
model.compile(loss='mse',
              optimizer='adam',
              metrics=['RootMeanSquaredError'])

# CREATE CALLBACKS
best_model_encdec_file=os.path.join(output_dir,'model_encdec.h5');
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_encdec_file, 
                                                save_weights_only=True,
                                                monitor='val_loss', 
                                                save_best_only=True, 
                                                verbose=1);

# Definindo o callback EarlyStopping
early_stopping = tf.keras.callbacks.EarlyStopping(  monitor='val_loss', 
                                                    patience=max(10,int(EPOCAS/10)),  
                                                    verbose=1, 
                                                    restore_best_weights=False);

log_dir = os.path.join(output_dir,"logs","fit",'encdec-'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"));
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# %%
# FIT THE MODEL
history = model.fit(train_data_generator,
                    epochs=EPOCAS,
                    validation_data=valid_data_generator,
                    callbacks=[checkpoint,early_stopping,tensorboard_callback],
                    verbose=1
                   );

# %%
mpp.save_model_history(history,
                       os.path.join(output_dir,"historical_encdec.csv"),
                       show=False,
                       labels=['root_mean_squared_error','loss']);

# %%
model.load_weights(best_model_encdec_file); 


best_model_encoder_file=os.path.join(output_dir,'model_encoder.h5');
best_model_decoder_file=os.path.join(output_dir,'model_decoder.h5');

model.layers[0].save_weights(best_model_encoder_file);
model.layers[1].save_weights(best_model_decoder_file);

# %%
"""
# Evaluate best encode-decoder model
"""

# %%
# LOAD BEST MODEL to evaluate the performance of the model
model.load_weights(best_model_encdec_file);
data_results=dict();

# Evaluate training
results = model.evaluate(train_data_generator)
results = dict(zip(model.metrics_names,results))
print('training',results,"\n\n");
for key,value in results.items():
    data_results['train_'+key]=value;

# Evaluate validation
results = model.evaluate(valid_data_generator)
results = dict(zip(model.metrics_names,results))
print('validation',results,"\n\n");
for key,value in results.items():
    data_results['val_'+key]=value;

# Evaluate testing
results = model.evaluate(test_data_generator)
results = dict(zip(model.metrics_names,results))
print('testing',results,"\n\n");
for key,value in results.items():
    data_results['test_'+key]=value;

data_results['number_of_parameters']=mpp.get_model_parameters(model);

# final all json
with open(os.path.join(output_dir,"training_data_results_encdec.json"), 'w') as f:
    json.dump(data_results, f,indent=4);

tf.keras.backend.clear_session()


# %%
"""
# Dataset
"""

# %%

training_dataset = sda.DataAugmentationClsGenerator(training_data_array, 
                                                    training_df.iloc[:,51].values,
                                                    batch_size=BATCH_SIZE, 
                                                    augment_fn=lambda X: slt.batch_random_rotate_coordinates(X,angle=ANGLE));

validation_dataset = sda.DataAugmentationClsGenerator(  validation_data_array, 
                                                        validation_df.iloc[:,51].values,
                                                        batch_size=BATCH_SIZE, 
                                                        augment_fn=None);

test_dataset = sda.DataAugmentationClsGenerator(test_data_array, 
                                                test_df.iloc[:,51].values,
                                                batch_size=BATCH_SIZE, 
                                                augment_fn=None);

# %%
"""
# Create new model
"""

# %%

model = mpp.create_model(load_weights=False,
                         file_of_weight=best_model_encoder_file,
                         file_of_weight_full=False,
                         ncod=NCOD);

print('input_shape',model.input_shape)
print('output_shape',model.output_shape)
print('')

model.summary()

mpp.save_model_parameters(model, os.path.join(output_dir,'parameters_stats_cls.m'));

# %%
# COMPILE NEW MODEL
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])

# CREATE CALLBACKS
best_model_file=os.path.join(output_dir,'model_cls.h5');
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_file, 
                                                save_weights_only=True,
                                                monitor='val_loss', 
                                                save_best_only=True, 
                                                verbose=1);

# Definindo o callback EarlyStopping
early_stopping = tf.keras.callbacks.EarlyStopping(  monitor='val_loss', 
                                                    patience=max(10,int(EPOCAS/10)),  
                                                    verbose=1, 
                                                    restore_best_weights=False);

log_dir = os.path.join(output_dir,"logs","fit",'cls-'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"));
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# %%
# FIT THE MODEL
history = model.fit(training_dataset,
                    epochs=EPOCAS,
                    validation_data=validation_dataset,
                    callbacks=[checkpoint,early_stopping,tensorboard_callback],
                    verbose=1
                   );

# %%
mpp.save_model_history(history,
                       os.path.join(output_dir,"historical_cls.csv"),
                       show=False,
                       labels=['categorical_accuracy','loss']);

# %%
# LOAD BEST MODEL to evaluate the performance of the model
model.load_weights(best_model_file);
data_results=dict();

# Evaluate training
results = model.evaluate(training_dataset)
results = dict(zip(model.metrics_names,results))
print('training',results,"\n\n");
for key,value in results.items():
    data_results['train_'+key]=value;

# Evaluate validation
results = model.evaluate(validation_dataset)
results = dict(zip(model.metrics_names,results))
print('validation',results,"\n\n");
for key,value in results.items():
    data_results['val_'+key]=value;

# Evaluate testing
results = model.evaluate(test_dataset)
results = dict(zip(model.metrics_names,results))
print('testing',results,"\n\n");
for key,value in results.items():
    data_results['test_'+key]=value;

data_results['number_of_parameters']=mpp.get_model_parameters(model);

# final all json
with open(os.path.join(output_dir,"training_data_results_cls.json"), 'w') as f:
    json.dump(data_results, f,indent=4);

tf.keras.backend.clear_session()

# %%
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

target_names = ['negative','neutral','pain','positive'];

# Predict
Y_pred = model.predict(test_dataset,verbose=1);
y_pred = np.argmax(Y_pred, axis=1);

# Calculate accuracy
categorical_accuracy = np.mean(test_dataset.classes == y_pred);
print(f'Categorical accuracy: {categorical_accuracy}');

# Confusion matrix

CM=confusion_matrix(test_dataset.classes, y_pred);

fname=os.path.join(output_dir,"confusion_matrix_cls.eps");
fig, ax = plt.subplots(figsize=(8,6), dpi=100)
disp = ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=target_names)
disp.plot(ax=ax,cmap=plt.cm.Blues)
plt.savefig(fname)

cm_dict=dict();
cm_dict['matrix']=CM.tolist();
cm_dict['label']=target_names;
# final all json
with open(os.path.join(output_dir,"confusion_matrix_cls.json"), 'w') as f:
    json.dump(cm_dict, f,indent=4);
    f.close()

# Classification report
fname=os.path.join(output_dir,"classification_report_cls.json")
dict_dat=classification_report(test_dataset.classes, y_pred, target_names=target_names,output_dict=True);
print(dict_dat)
with open(fname, 'w') as f: 
    json.dump(dict_dat, f,indent=4);
    f.close()