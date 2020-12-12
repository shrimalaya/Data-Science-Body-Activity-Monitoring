# CMPT353 PROJECT : BODY ACTIVITY MONITORING
> In this project, we predict a user's activity based on whether they are standing, walking or running.

#### _A Report with all the analysis and  details of the used techniques is saved as the file Report.pdf_

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Libraries Used](#libraries)
* [Features](#features)
* [Status](#status)
* [Based On](#basis)
* [Created By](#creation)

## General info
The idea behind this project is to predict based on a user’s activity whether they are standing, walking, or running using machine learning algorithms.Data was collected through sensors and data preprocessing was done. Statistical tests were used to analyze the data. Feature engineering was used for training Machine Learning Models.


## Technologies
You will need to install Anaconda.
* Python - version 3.0

## Libraries Used
numpy
pandas
matplotlib.pyplot
statsmodels.nonparametric.smoothers_lowess 
scipy 


## Features
List of features ready 
* Data preprocessing
* Statistics
* Machine Learning

## Order Of Execution 
         1) Run 'analysis.ipynb'
            - Results produced: filtered files for each scenario
      
                
        2) Run 'Statistics.ipynb'
            - Results produced: one transformed file and graphs of multiple inferential and statistical tests
              
            
        3) Run 'Machine Learning.ipynb'
            - Results produced:Classification report, ROC Curve and Confusion matrix for each model 
            - Models saved in location: "Models"
            
        4) Run 'Prediction.ipynb'
            - Imports Saved models from "Models" 
           - Predicts and print results for never seen data in "Data/testData"
     

## Status
Project is: _finished_

## Based on
Project based on Computational Datascience

## Created By
Created by [Arpit Kaur,Sharjeel Ahmed and Srimalaya Ladha]
