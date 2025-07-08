import os

# basic data engineering
import pandas as pd
import numpy as np
import scipy

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# db
import pymongo

# configs & other
import yaml
from tqdm.notebook import tqdm_notebook
from datetime import datetime
from time import time

from psynlig import pca_explained_variance_bar

# utils processing
from utils import sliding_window_pd
from utils import apply_filter
from utils import filter_instances
from utils import flatten_instances_df
from utils import df_rebase
from utils import rename_df_column_values
from utils import encode_labels

# utils visualization
from utils_visual import plot_instance_time_domain
from utils_visual import plot_instance_3d
from utils_visual import plot_np_instance
from utils_visual import plot_heatmap
from utils_visual import plot_scatter_pca

# training
from sklearn.model_selection import train_test_split

# scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# add transformers
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

time_start = time()

config_path = os.path.join(os.getcwd(), "config.yml")

with open(config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

client = pymongo.MongoClient(config["client"])

db = client[config["db"]]
coll = db[config["col"]]

found_keys = coll.distinct("label")
print("Existing DB keys:", found_keys)

documents = list(coll.find())
dfs = []
for document in documents:
    df_doc = pd.DataFrame(document["data"])
    df_doc["label"] = document["label"]
    dfs.append(df_doc)

df = pd.concat(dfs, ignore_index=True)

order_list = list(documents[0]['data'].keys()) + ['label']
ref_list = order_list.copy()

df = df_rebase(df, order_list, ref_list)
# print(df)
# apply_filter

signal_columns = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

X = df[signal_columns].to_numpy()

df[signal_columns]= apply_filter(X, order=4, wn=0.1, filter_type="lowpass")

print(df)
#plot_instance_time_domain(df)

# sliding_window_pd
windows = sliding_window_pd(df, ws=20, overlap=20)
'''''
for i, window in enumerate(windows):
    print(f"--- Window {i+1} ---")
    print(window)
    print("\n")
'''''
window_labels = []

for window in windows:
   
    labels_in_window = window["label"].values
    
    unique, counts = np.unique(labels_in_window, return_counts=True)
    majority_label = unique[np.argmax(counts)]
    window_labels.append(majority_label)

#print(window_labels)

# filter_instances
signal_only_windows = [w[signal_columns] for w in windows]
filtered_signal_only = filter_instances(signal_only_windows, order=4, wn=0.1, filter_type="lowpass")
#print(filtered_signal_only)

# flatten_instances_df
flattened_df = flatten_instances_df(filtered_signal_only)
print(flattened_df)

# Labels
y = df["label"]
#print(y)

# rename_df_column_values
final_df = rename_df_column_values(flattened_df.to_numpy(), window_labels, flattened_df.columns.tolist())
print(final_df.iloc[:, -1].unique())

X= final_df.iloc[:, :-1]
y= final_df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=42)
#print(y_train)
#print(X_train)
#print(y_train.unique())

# Scaling
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA fitting
pca2d = PCA(n_components=3)  # for 3D visualization
pca2d.fit(X_train_scaled)

# Transform scaled data
X_train_pca = pca2d.transform(X_train_scaled)
X_test_pca = pca2d.transform(X_test_scaled)

# Variance diagramm
pca_explained_variance_bar(pca2d, alpha=0.8)

X_train_pca_df = pd.DataFrame(X_train_pca, columns=["PC1", "PC2", "PC3"])
X_train_pca_df["label"] = y_train.reset_index(drop=True)
#print(X_train_pca_df.columns)

# Visualization
plot_scatter_pca(X_train_pca_df, c_name="label")

y_train=encode_labels(y_train)
y_test =encode_labels(y_test)

# 1. SVC Classifier
svc = SVC(kernel='rbf', C=1, gamma='scale')

svc.fit(X_train_pca, y_train)
y_pred_svc = svc.predict(X_test_pca)

print("=== SVC Classification Report ===")
print(classification_report(y_test, y_pred_svc))

cm_svc = confusion_matrix(y_test, y_pred_svc, labels=svc.classes_)
disp_svc = ConfusionMatrixDisplay(confusion_matrix=cm_svc, display_labels=svc.classes_)
disp_svc.plot()
plt.title("SVC Confusion Matrix")
plt.show()


# 2. Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_pca, y_train)
y_pred_rf = rf.predict(X_test_pca)

print("=== Random Forest Classification Report ===")
print(classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf, labels=rf.classes_)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=rf.classes_)
disp_rf.plot()
plt.title("Random Forest Confusion Matrix")
plt.show()

# 3. GridSearchCV for SVC
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train_pca, y_train)

print("=== Best parameters from GridSearchCV ===")
print(grid_search.best_params_)

# Evaluate best estimator
best_svc = grid_search.best_estimator_
y_pred_best_svc = best_svc.predict(X_test_pca)

print("=== Best SVC Classification Report ===")
print(classification_report(y_test, y_pred_best_svc))

cm_best = confusion_matrix(y_test, y_pred_best_svc, labels=best_svc.classes_)
disp_best = ConfusionMatrixDisplay(confusion_matrix=cm_best, display_labels=best_svc.classes_)
disp_best.plot()
plt.title("Best SVC (GridSearch) Confusion Matrix")
plt.show()


