#!/usr/bin/python

from setuptools import setup, find_packages
import os 

def func_read(fname):
    return open(os.path.join(os.path.dirname(__file__),fname)).read();

setup(
    name   ='SkeletonEmotion4Lib',
    version='0.1.0',
    author='Fernando Pujaico Rivera',
    author_email='fernando.pujaico.rivera@gmail.com',
    maintainer='Fernando Pujaico Rivera',
    maintainer_email='fernando.pujaico.rivera@gmail.com',
    #scripts=['bin/script1','bin/script2'],
    url='https://github.com/trucomanx/fcnn_emotion4',
    license='GPLv3',
    description='Functions to detect 4 emotions',
    #long_description=func_read('README.md'),
    include_package_data=True,
    packages=['SkeletonEmotion4Lib'], #find_packages(where="SkeletonEmotion4Lib"),
    package_dir={'SkeletonEmotion4Lib':'SkeletonEmotion4Lib'},
    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        #'': ['*.py','*.h5'],
        'SkeletonEmotion4Lib': ['*.py','models/*.h5']
    },
    install_requires=[ #"Django >= 1.1.1",
       "tensorflow",
       "tensorflow-hub",
       "opencv-python", 
       "matplotlib",
       "numpy",
       "nvidia-cudnn-cu11",
       "tf-keras"
    ],
)

#! python setup.py sdist bdist_wheel
# Upload to PyPi
# or 
#! pip3 install dist/SkeletonEmotion4Lib-0.1.0.tar.gz 
