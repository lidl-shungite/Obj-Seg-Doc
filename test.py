import pandas as pd
from sklearn.metrics import confusion_matrix

df = pd.read_csv("logs/yolo_testing_data.csv")
true = list(df["y_true"])
pred = list(df["y_pred"])
print(confusion_matrix(y_true=true,y_pred=pred).shape)