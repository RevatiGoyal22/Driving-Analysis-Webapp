from datetime import datetime
from random import random

from django.http import JsonResponse
from django.shortcuts import render
import numpy as np
import seaborn as sns
import pandas as pd

data = pd.read_excel('C:/Users/91638/PycharmProjects/pythonProject1/proj/Train_test_data_for_app.xlsx')
df_test = pd.read_excel("C:/Users/91638/PycharmProjects/pythonProject1/proj/Train_test_data_for_app.xlsx", "Sheet2")

data_new = data[['ibi', 'rmssd', 'sd1/sd2', 'lf/hf', 'Cluster']]
X = data_new.iloc[:, 0:4]
Y = data_new.iloc[:, -1]
df_test_new = df_test[['ibi', 'rmssd', 'sd1/sd2', 'lf/hf', 'Cluster']]
X_test = df_test_new.iloc[:, 0:4]
Y_test = df_test_new.iloc[:, -1]
Y_test = Y_test.astype(int)

from sklearn.cluster import KMeans

wcss = []

for i in range(1, 12):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

sns.set()

kmeans = KMeans(n_clusters=3, init='k-means++')
Y_pred = kmeans.fit_predict(X)

Y_test = kmeans.fit_predict(X_test)

data_new_agg = data[['Speed (km/h)', 'Longitudinal acceleration (m/sÂ²)']]
X1 = data_new_agg.iloc[:, 0:2]
Y1 = data_new_agg.iloc[:, -1]
df_test_new_agg = df_test[['Speed (km/h)', 'Longitudinal acceleration (m/sÂ²)']]
X_test1 = df_test_new_agg.iloc[:, 0:2]
Y_test1 = df_test_new_agg.iloc[:, -1]
Y_test1 = Y_test1.astype(int)

wcss = []

for i in range(1, 12):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(X1)

    wcss.append(kmeans.inertia_)

sns.set()

kmeans = KMeans(n_clusters=3, init='k-means++')
Y_pred_agg = kmeans.fit_predict(X1)

X_bar = ["Aggressive", "Neutral", "Conservative"]
count_1 = np.sum(Y_pred_agg == 1)
count_2 = np.sum(Y_pred_agg == 2)
count_0 = np.sum(Y_pred_agg == 0)
total = count_0 + count_1 + count_2
count_1_perc = count_1 * 100 / total
count_2_perc = count_2 * 100 / total
count_0_perc = count_0 * 100 / total
Y_bar = [count_2_perc, count_0_perc, count_1_perc]

min_0, max_0, min_1, max_1, min_2, max_2 = 10000, -10000, 10000, -10000, 10000, -10000
for i in range(len(data["Speed (km/h)"])):
    if Y_pred[i] == 0:
        if data["Speed (km/h)"][i] < min_0:
            min_0 = data["Speed (km/h)"][i]
        if data["Speed (km/h)"][i] > max_0:
            max_0 = data["Speed (km/h)"][i]
    if Y_pred[i] == 1:
        if data["Speed (km/h)"][i] < min_1:
            min_1 = data["Speed (km/h)"][i]
        if data["Speed (km/h)"][i] > max_1:
            max_1 = data["Speed (km/h)"][i]
    if Y_pred[i] == 2:
        if data["Speed (km/h)"][i] < min_2:
            min_2 = data["Speed (km/h)"][i]
        if data["Speed (km/h)"][i] > max_2:
            max_2 = data["Speed (km/h)"][i]

categories = ["Aggressive", "Neutral", "Conservative"]
stress_bar = [[min_2, max_2], [min_1, max_1], [min_0, max_0]]


avgStress = 0
for i in Y_pred:
    avgStress = avgStress+Y_pred[i]
avgStress = avgStress/len(Y_pred)

avgAgg = 0
for i in Y_pred_agg:
    avgAgg = avgAgg+Y_pred_agg[i]
avgAgg = avgAgg/len(Y_pred_agg)
bpm = data["bpm"].values.tolist()
ibi = data["ibi"].values.tolist()
speed = data["Speed (km/h)"].values.tolist()


def result_view(request):
    return render(request, 'result.html', {"Agg_X": X_bar, "Agg_Y": Y_bar, "stress_bar": stress_bar,
    "avgAgg": avgAgg, "avgStress": avgStress})


def real_time_view(request):
    return render(request, 'real_time.html', {"bpm": bpm, "speed": speed, "ibi": ibi})

