# Machine learning development

This repo contains scripts to import GALFORM data, train and evaluate a machine learning emulator using the GALFORM data, and finally calibrate a new GALFORM model on observations using the emulator and Bayesian statistical methods. 

## Required Python Packages 

As well as the usual scientific packages (e.g. NumPy, Pandas, matplotlib etc...) it is a requirement to have the Tensorflow and Keras packages installed. 
For example, if using pip: 

```bash
pip install tensorflow
```
This should also install Keras within TensorFlow. You will see how Tensorflow and Keras are used in the scripts. 

## Relevant scripts

This repository contains many project-specific scripts that are not required for the emulation. 
I suggest the main Python scripts to focus on are: 

- Data Generation
  - Formating the training data for the emulator training and calibration
- MultiOut_ML
  - Creating the emulator ensemble model
- model_eval_traintest
  - Testing the emulator performance using the testing data set
- Optimization
  - Running the MCMC to fit the emulator to the observation data
