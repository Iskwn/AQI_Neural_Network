# Air-Quality-Neural-Network

## Overview

A program that seeks to predict the future Air Quality values in major international cities using historical air quality data and machine learning

## Motivation / Reason

I decided to build this project to get a better understanding of how Machine Learning, particularly Neural Networks, function. The need for a Science Fair Project for my science class in School is what caused me to actually begin this project (plus I think it would be good for Science Fair). This project will also allow me to learn more about PyTorch (which I've known existed for some time now just never explored up until this point)

## Progress

### COMPLETED

- Basic Network Completed
- PyTorch Implementation started

### IN PROGRESS

- Data Collection from OpenAQ
- Find Suitable Network Structure

### PLANNED

- Update Network to the suitable structure
- Train Model and Evaluate Outputs
- Improve Network based on weaknessed identified in previous step
- Record Results with Matplotlib

## Technical Details

-Framework: PyTorch  
-Architecture: Basic 3-layer feedforward neural network [Will Later Transition to a more appropriate stucture]  
-Dataset: OpenAQ Air Quality measurements  
-Current Features:  
    - Historical PM2.5 Levels
    - Temperature
    - Time Features (day, season)
-Future Features:  
    - Additional Pollutants( NO2, O3, PM10)
    - Wind Speed and Direction
    - Humidity
-Target Cities: Tokyo, Moscow, Berlin, Quebec, Warsaw, Venice, Paris, London, Delhi, Beijing

## Challenges

Overall Challenges during the Building Process:  
Finding the optimal model type (10/9/25)  
Append as necessary during Building

## Future Work

- Add more options for Predictions (e.g. predict PM2.5, PM10, O3, and NO2 instead of just PM2.5)  
- Create a Web Interface for the Model (Likely via Steamlit)

## Program Requirements

All Library Versions are not hard requirements, they are just the versions I am using at the time of creation.  
-Pytorch 2.8.0 or newer  
    [Pytorch Installer](https://pytorch.org)  
-Numpy 2.3.2 or newer  
    [Numpy Installer](https://numpy.org/install/)  
-Pandas 2.3.3 or newer  
    [Pandas Installer](https://pandas.pydata.org/docs/install.html)  
-Git 2.51.0 or newer  
    [Git Installer](https://git-scm.com/downloads)  

## Usage Instructions

Instructions will be tested for accuracy when program is completed  

Clone the Github Repository :  
    - `git clone https://github.com/Iskwn/Air-Quality-Neural-Network.git`  

Travel to new Repo with CD:  
    - `cd ./Air-Quality-Neural-Network`  

Install Necessary Libraries:  
    - `refer to Program Requirements for install links, follow the instructions on the sites.`  

etc

## Author

Jariel Reno - Junior in HS  
Science Fair Project, 2025

## Acknowledgements

- `OpenAQ for Air Quality Data`
- `PyTorch Team for the Model's Framework`
- [Kavan](https://github.com/kavan010) `for github repository setup references / formatting`
