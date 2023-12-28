# **League of Legends Win Analysis Project**
A comprehensive deep learning project on binary classification (predict winning chance of a team). 
## Project Summary
Based on the Kaggle data with almost 10k ranked League games stats at the 10 minute mark, I trained a MLP model achieving 73% accuracy on predicting winning chances of a team.  
Then I generated sequence data for each game (0-9 minute stat) based on my in game domain knowlege and trained a LSTM model achieving a bit higher accuracy which is around 73.1%.
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
