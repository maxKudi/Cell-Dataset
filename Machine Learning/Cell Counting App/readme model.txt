#Cell Counting Neural Network

#How to Use (You can also refer to the video included in this folder for usage of cell count)
1. Make sure you have python(recommended python 3.5-3.7) installed on your computer. The required packages(see below) and modules also need to be properly installed prior to using the software.
2. Rename your cells' images folder directory into original_images.
3. Make sure that your data is stored with the proper directory i.e. .../Original image(16bit)/A1/yourimagename.png , .../Original image(16bit)/A2/yourimagename.png or else the program won't work
4. Open the test_images folder and remove the previous set of images if you want to run the new set of images
5. Copy the main folder of your datas i.e. Original image(16bit) into the test_images folder in this directory and rename the Original image(16bit) into original_image, make sure your image folder directory look like this .../original_images/
6. You can run your cell counting on your command prompt/terminal or in the python shell(IDLE).
7. To run on cmd or terminal:
    -Open your cmd or your terminal and cd into this folder. e.g. cd C:/Users/user1/Desktop/cell_counting_group1
    -Enter this command : python object_detection_counting.py
8. To run on python shell(IDLE):
    -Open your python shell(this should be included with the installation of python)
    -Click on File(top left) and click(ctrl + o for shortcut) open and select object_detection_counting.py in the designated folder
    -You will object_detection_counting script pop up on new window.
    -Click Run or F5 for shortcut
9. Two output will be produced namely result.txt and final_result.txt

###################

#Required Modules:
- pillow
- lxml
- cython
- contextlib2
- matplotlib
- pandas
- opencv-python
- tensorflow or tensorflow-gpu
1. Check modules installed by running this command on command prompt. command -> "pip list"
2. If these modules have not been installed please install it. For windows use pip install command.


Notes: you can email our team if encounter any problem

