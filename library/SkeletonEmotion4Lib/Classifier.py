#!/usr/bin/python

import os
import SkeletonEmotion4Lib.lib_model as mpp

class Emotion4Classifier:
    """Class to classify 4 body languages.
    
    The class Emotion4Classifier classify daa in 4 body languages.
    
    Args:
        :param file_of_weight: Archivo donde se encuentran los pesos.
        
    Atributos:
        modelo: Model returned by tensorflow.
    """
    def __init__(self,file_of_weight=''):
        """Inicializer of class Emotion4Classifier.
        
        Args:
            param file_of_weight: Archivo donde se encuentran los pesos.
        """
        
        if len(file_of_weight)>0:
            self.model = mpp.create_model_encoder(  load_weights=False,
                                                    file_of_weight=file_of_weight);
        else:
            self.model = mpp.create_model_encode(   load_weights=True,
                                                    file_of_weight='');

    def from_skel_npvector(self,npvector):
        """Classify a skeleton data from a numpy vector object.
        
        Args:
            npvector: Numpy vector.
        
        Returns:
            int: The class of image.
        """
        return mpp.evaluate_model_from_npvector(self.model,npvector);


    def target_labels(self):
        """Returns the categories of classifier.
        
        Returns:
            list: The labels of categories resturned by the methods from_skeleton_npvector().
        """
        return ['negative','neutro','pain','positive'];


