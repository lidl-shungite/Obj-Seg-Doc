import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import random

df = pd.read_csv("logs/1st_logs.csv")
new_df = df[["epoch","mean_squared_error","val_mean_squared_error"]]
new_df.to_csv("logs/mse_log.csv", index=False)