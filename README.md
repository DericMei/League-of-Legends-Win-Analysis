# **League of Legends Win Analysis Project**
A comprehensive deep learning project on binary classification (predict winning chance of a team). 
## Project Summary
Based on the Kaggle data with almost 10k ranked League game stats at the 10 minute mark, I was able to train both a MLP model and generate data to train a LSTM model to predict the winning chances of a team based on the given stats at the 10 minute mark. Best accuracy performance of 73.38% was achieved by stacking the results of both models and run logistic regression on the results, which is a way to combine model results.
## Project Approach Pipeline
- Setting Objectives for the project
  - Binary Classification
  - Need sequence data for LSTM
- Data Preparation/Understanding
  - Data Cleaning
    - Basic data cleaning
    - Normalization of numerical columns
  - EDA
    - Sweetviz report
    - Comparing different columns by visualization
    - Unsupervised Learning
      - Factor Analysis
      - Cluster Analysis
    - Correlation Matrix
  - Generated Sequence Data (using domain knowledge)
- Data Splitting/Loading/Batching
  - Training/Validation/Testing data splitting (80%/10%/10%)
  - Loaded data into tensor datasets and batched them
- Modeling
  - Logistic regression as benchmark
  - MLP model training
    - Manually tuned hyper-parameters to find best range
    - Bayesian optimization, 50 trails
    - Manually tuned the results again
  - LSTM model training
    - Only manually
  - Combine model results
    - Averaging
    - Weighted average
    - Stacking
    - Model blending
  - Final comparison of model performance
- Model Interpretation/Evaluation
  - Feature Importance (Permuation importance)
  - Confusion Matrix
  - ROC Curve (AUC)
- Deployment/Conclusions
## Key Take-aways
- By interpreting results from factor and cluster analysis, it is very interesting to see that there are 43.7% of games that one team is dominating the other at the 10 minute mark, which indicates that there are 56.7% games that still could be turned around at the 10 minute mark.
- Although both of my deep learning methods beat the benchmark, it was only by a tiny little bit. I can’t say that I am not disappointed. I think that the reason behind is that I do not have enough data. Deep learning models are way too data hungry, and my data does not even contain 10 thousand rows, which could be the main reason of deep learning methods not performing significantly better. Another reason could be because that my task is a very simple binary classification task, and Logistic Regression really excels at this task. One other interesting finding I have here is that my LSTM model performed the best out of the 3 models, with the data I generated! This is very surprising to me, and I suspect the main reason behind is that I really applied my deep domain knowledge here to generate data so that the data I generated are not garbage. They could mimic what real data looks like! Overall, combining results from the 2 models worked, but not that much, the reason behind could be they are based on very similar data.    
- Although it is pretty hard to interpret deep learning model features, I was able to figure out permuation importance to interpret the results from both of my models. It turns out that besides obvious ones, dragons or elite monsters overall are very important features even for at the 10 minute mark. This is a very interesting finding for me because people tend not to care about these neutral objectives in game that early (at the 10 minute mark).    
- Overall, this was a very comprehensive data science project that covered various techniques. Although I was hoping for a higher accuracy, I am still pretty happy with the result I have. With what I have at this moment, I could potentially link my LSTM model to a live ranked game, and predict the winning chances of a team as the game goes on.
## Tools Used
Python, Jupyter Notebook
## Data Science Techniques Used
Pandas, Numpy, PyTorch, EDA, Logestic Regression, MLP, LSTM, Bayesian Optimization, Permutation Importance, Seaborn, Matplotlib, Unsupervised Learning, Cluster Analysis, Factor Analysis, Supervised Machine Learning, etc.
## Files
- Project Code.ipynb (Jupyter notebook file that contains all the codes for the project)
- high_diamond_ranked_10min.csv (raw data from Kaggle)
- LOL win Analysis.pptx (ppt file I made to present the project)
- model_lstm.pth (best trained LSTM model path saved)
- model_mlp.pth (best trained MLP model path saved)
- Project Report.docx (the project report in word file)
- Project Report.pdf (the project report in pdf file)
- SWEETVIZ_REPORT.html (the file sweetviz generated for EDA)
- plots (A folder that includes plots files that are used in the report)
## Usage
The project was originally coded on Google Colab.
## Data Source
The raw data used in this project was from [Kaggle](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min).
