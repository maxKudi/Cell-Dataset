#This script is used to install the required package for you directly
#if you run this script there's no need to install each packages individually
#run this on cmd -> python setup.py

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m","pip", "install", package])

def install2(package):
    subprocess.check_call([sys.executable, "-m","pip", "install", "--user", package])

install("tensorflow")
install("pillow")
install("lxml")
install("cython")
install("pandas")
install("opencv-python")
install("matplotlib")
install("contextlib2")
install("pycocotools")
install ("PyQt5")
install ("tensorflow-object-detection-api")