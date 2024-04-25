from datetime import datetime
from random import random

from django.http import JsonResponse
from django.shortcuts import render
import numpy as np
import seaborn as sns
import pandas as pd

# Data Import
data = pd.read_excel('./Train_test_data_for_app.xlsx')
data_previous = pd.read_excel('./previous_data.xlsx')

data_new = data[['ibi', 'rmssd', 'sd1/sd2', 'lf/hf', 'Cluster']]
X = data_new.iloc[:, 0:4]
Y = data_new.iloc[:, -1]

# Model Building

from sklearn.cluster import KMeans

wcss = []

for i in range(1, 12):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

sns.set()

kmeans = KMeans(n_clusters=3, init='k-means++')
Y_pred = kmeans.fit_predict(X)

data_new_agg = data[['Speed (km/h)', 'Longitudinal acceleration (m/sÂ²)']]
X1 = data_new_agg.iloc[:, 0:2]
Y1 = data_new_agg.iloc[:, -1]

wcss = []

for i in range(1, 12):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(X1)

    wcss.append(kmeans.inertia_)

sns.set()

kmeans = KMeans(n_clusters=3, init='k-means++')
Y_pred_agg = kmeans.fit_predict(X1)

df_test = pd.read_excel("./Train_test_data_for_app.xlsx", "Sheet2")
speed = df_test["Speed (km/h)"].values.tolist()

df_test_new = df_test[['ibi', 'rmssd', 'sd1/sd2', 'lf/hf', 'Cluster']]
X_test = df_test_new.iloc[:, 0:4]
Y_test = df_test_new.iloc[:, -1]
Y_test = Y_test.astype(int)

df_test_new_agg = df_test[['Speed (km/h)', 'Longitudinal acceleration (m/sÂ²)']]
X_test1 = df_test_new_agg.iloc[:, 0:2]
Y_test1 = df_test_new_agg.iloc[:, -1]
Y_test1 = Y_test1.astype(int)

df_prev_new = data_previous[['ibi', 'rmssd', 'sd1/sd2', 'lf/hf']]
X_prev = df_prev_new.iloc[:, 0:4]

df_prev_new_agg = data_previous[['Speed (km/h)', 'Longitudinal acceleration (m/sÂ²)']]
X_prev1 = df_prev_new_agg.iloc[:, 0:2]

Y_pred_test = kmeans.fit_predict(X_test).tolist()
Y_pred_agg_test = kmeans.fit_predict(X_test)
Y_pred_agg_test_send = Y_pred_agg_test.tolist()


def real_time_view(request):
    return render(request, 'real_time.html', {"speed": speed, "Y_pred_stress": Y_pred_test,
                                              "Y_pred_agg": Y_pred_agg_test_send})


X_bar = ["Aggressive", "Neutral", "Conservative"]
count_1 = np.sum(Y_pred_agg_test == 1)
count_2 = np.sum(Y_pred_agg_test == 2)
count_0 = np.sum(Y_pred_agg_test == 0)
total = count_0 + count_1 + count_2
count_1_perc = count_1 * 100 / total
count_2_perc = count_2 * 100 / total
count_0_perc = count_0 * 100 / total
Y_bar = [count_1_perc, count_2_perc, count_0_perc]

min_0, max_0, min_1, max_1, min_2, max_2 = 10000, -10000, 10000, -10000, 10000, -10000
highStress = 0
for i in range(len(df_test["Speed (km/h)"])):
    if Y_pred[i] == 0:
        if df_test["Speed (km/h)"][i] < min_0:
            min_0 = data["Speed (km/h)"][i]
        if df_test["Speed (km/h)"][i] > max_0:
            max_0 = df_test["Speed (km/h)"][i]
    if Y_pred[i] == 1:
        if df_test["Speed (km/h)"][i] < min_1:
            min_1 = df_test["Speed (km/h)"][i]
        if df_test["Speed (km/h)"][i] > max_1:
            max_1 = df_test["Speed (km/h)"][i]
            highStress = highStress + 1
    if Y_pred[i] == 2:
        if df_test["Speed (km/h)"][i] < min_2:
            min_2 = df_test["Speed (km/h)"][i]
        if df_test["Speed (km/h)"][i] > max_2:
            max_2 = df_test["Speed (km/h)"][i]
highStress = highStress * 100 / len(df_test["Speed (km/h)"])

stress_bar = [[min_1, max_1], [min_0, max_0], [min_2, max_2]]
avgAgg = 0
for i in range(len(Y_pred_agg_test)):
    avgAgg = avgAgg + Y_pred_agg_test[i]
avgAgg = avgAgg / len(Y_pred_agg_test)
avgStress = 0
for i in range(len(Y_pred_test)):
    avgStress = avgStress + Y_pred_test[i]
avgStress = avgStress / len(Y_pred_test)

# Previous data
y_prev_pred_stress = kmeans.fit_predict(X_prev).tolist()
y_prev_pred_agg = kmeans.fit_predict(X_prev1).tolist()

highStressPrev = 0
noRiskyPrev = 0
for i in range(len(y_prev_pred_stress)):
    if y_prev_pred_stress[i] == 1:
        highStressPrev = highStressPrev + 1
    risk = ((0.6 * y_prev_pred_agg[i]) + (0.4 * y_prev_pred_stress[i])) * 50
    if risk > 66:
        noRiskyPrev = noRiskyPrev + 1
percRiskyPrev = noRiskyPrev * 100 / len(y_prev_pred_stress)
highStressPrev = highStressPrev * 100 / len(y_prev_pred_stress)

highAggPrev = 0
for i in range(len(y_prev_pred_agg)):
    if y_prev_pred_agg[i] == 1:
        highAggPrev = highAggPrev + 1
highAggPrev = highAggPrev * 100 / len(y_prev_pred_agg)
diff_agg = count_1_perc - highAggPrev
stress_diff = highStress - highStressPrev
noTeaBreakAlert = 0
alert = 0

for i in range(len(Y_pred_test)):
    if i % 50 == 0:
        if alert == 50:
            noTeaBreakAlert = noTeaBreakAlert + 1
        alert = 0
        if Y_pred_test[i] == 1:
            alert = alert + 1
    else:
        if Y_pred_test[i] == 1:
            alert = alert + 1
print(noTeaBreakAlert)


def result_view(request):
    return render(request, 'result.html',
                  {"Agg_X": X_bar, "Agg_Y": Y_bar, "stress_bar": stress_bar, "avgAgg": avgAgg,
                   "avgStress": avgStress, "diffAgg": diff_agg, "diffStress": stress_diff,
                   "diffRisky": percRiskyPrev, "alerts": highStress, "teaBreakAlert": noTeaBreakAlert})
