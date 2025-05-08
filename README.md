# README

### Retina U-Net Project

This is an attempted reproduction of the Retina U-Net: Embarrassingly Simple Exploitation of Segmentation Supervision for Medical Object Detection paper. 
It creates a lightweight Retina U-Net model with a relatively simple RetinaNet head for learning bounding boxes and a U-Net pixel wise segmentation
decoder. The original paper's github repo can be found here: https://github.com/MIC-DKFZ/medicaldetectiontoolkit.


### How to run

clone the repository with

git clone https://github.com/BrendanJeaney/BrendanHeaney_DL4H_Project.git

Then create a virtual environment and install the dependencies

python -m venv venv

venv/Source/activate   # For windows, or source venv/bin/activate for linux/mac

pip install -r requirements.txt


Run the project_testing.py file to build the U-Net and Retina U-Net models

python project_testing.py



### If you run into issues running the code

- Check that the proper dependecies are installed
- Check you are using Python 3.10.x (or 3.8.x or 3.9.x)
- Make sure you are passing the U-Net the correct data filepath
- Ensure you are building the U-Net and passing it to the Retina U-Net
