# Packaging


Download the source code

    
    git clone https://github.com/trucomanx/fcnn_emotion4

The next command generates the `dist/SkeletonEmotion4Lib-VERSION.tar.gz` file.

    cd fcnn_emotion4/library
    python3 setup.py sdist

For more informations use `python setup.py --help-commands`

# Install 

Install the packaged library

    pip3 install dist/SkeletonEmotion4Lib-*.tar.gz

# Uninstall

    pip3 uninstall SkeletonEmotion4Lib
