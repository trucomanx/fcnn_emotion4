# fcnn_emotion4
fcnn_emotion4

# Using library
Since the code uses an old version of keras, it needs to be placed at the beginning of the main.py code.

    import os
    os.environ['TF_USE_LEGACY_KERAS'] = '1'

# Installation summary

    git clone https://github.com/trucomanx/fcnn_emotion4
    cd fcnn_emotion4/library
    python3 setup.py sdist
    pip3 install dist/SkeletonEmotion4Lib-*.tar.gz



