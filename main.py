from joblib import load
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy import signal


def loadData():
    walkDF = []
    standDF = []
    runDF = []

    for i in range(4):
        walkDF.append(pd.read_csv("data/testdata/walk{}.csv".format(i + 1)))
        standDF.append(pd.read_csv("data/testdata/stand{}.csv".format(i + 1)))
        runDF.append(pd.read_csv("data/testdata/run{}.csv".format(i + 1)))

    return standDF, walkDF, runDF


def loadModels():
    # # Model Load
    rfModel = load('models/random_forest.joblib')
    mlpModel = load('models/neural_network.joblib')
    svcModel = load('models/svc_classifier.joblib')

    return rfModel, mlpModel, svcModel


def cleanData(standDF, walkDF, runDF):
    # using low pass filter
    b, a = signal.butter(3, 0.1, btype='lowpass', analog=False)

    walk_np = list(walkDF)
    standing_np = list(standDF)
    running_np = list(runDF)

    butter_worth_walk = []
    butter_worth_standing = []
    butter_worth_running = []

    for i in range(4):
        butter_worth_walkX = signal.filtfilt(b, a, walk_np[i]["gFx"])
        butter_worth_walkY = signal.filtfilt(b, a, walk_np[i]["gFy"])
        butter_worth_walkZ = signal.filtfilt(b, a, walk_np[i]["gFz"])
        butter_worth_walk.append((pd.DataFrame(list(
            zip(walk_np[i]['time'], butter_worth_walkX, butter_worth_walkY, butter_worth_walkZ, walk_np[i]['acc'])),
                                               columns=["Time", "AccX", "AccY", "AccZ", "Acc-old"])))

        butter_worth_standingX = signal.filtfilt(b, a, standing_np[i]["gFx"])
        butter_worth_standingY = signal.filtfilt(b, a, standing_np[i]["gFy"])
        butter_worth_standingZ = signal.filtfilt(b, a, standing_np[i]["gFz"])
        butter_worth_standing.append((pd.DataFrame(list(
            zip(standing_np[i]['time'], butter_worth_standingX, butter_worth_standingY, butter_worth_standingZ,
                standing_np[i]['acc'])),
                                                   columns=["Time", "AccX", "AccY", "AccZ", "Acc-old"])))

        butter_worth_runningX = signal.filtfilt(b, a, running_np[i]["gFx"])
        butter_worth_runningY = signal.filtfilt(b, a, running_np[i]["gFy"])
        butter_worth_runningZ = signal.filtfilt(b, a, running_np[i]["gFz"])
        butter_worth_running.append((pd.DataFrame(list(
            zip(running_np[i]['time'], butter_worth_runningX, butter_worth_runningY, butter_worth_runningZ,
                running_np[i]['acc'])),
                                                  columns=["Time", "AccX", "AccY", "AccZ", "Acc-old"])))

    for i in range(4):
        butter_worth_walk[i]["Acc-trns"] = np.sqrt(butter_worth_walk[i]["AccX"] ** 2 *
                                                   butter_worth_walk[i]["AccY"] ** 2 *
                                                   butter_worth_walk[i]["AccZ"] ** 2)

        butter_worth_standing[i]["Acc-trns"] = np.sqrt(butter_worth_standing[i]["AccX"] ** 2 *
                                                       butter_worth_standing[i]["AccY"] ** 2 *
                                                       butter_worth_standing[i]["AccZ"] ** 2)

        butter_worth_running[i]["Acc-trns"] = np.sqrt(butter_worth_running[i]["AccX"] ** 2 *
                                                      butter_worth_running[i]["AccY"] ** 2 *
                                                      butter_worth_running[i]["AccZ"] ** 2)

        # Update current data with cleaned data
    newWalk = []
    newStand = []
    newRun = []

    for i in range(4):
        newWalk.append(butter_worth_walk[i])
        newStand.append(butter_worth_standing[i])
        newRun.append(butter_worth_running[i])

    return newStand, newWalk, newRun


def applyStatistics(df):
    # ### Finding mean, min and max

    df1 = pd.DataFrame(df[0][['AccX', 'AccY', 'AccZ', 'Acc-old', 'Acc-trns']].mean()).T
    df2 = pd.DataFrame(df[1][['AccX', 'AccY', 'AccZ', 'Acc-old', 'Acc-trns']].mean()).T
    df3 = pd.DataFrame(df[2][['AccX', 'AccY', 'AccZ', 'Acc-old', 'Acc-trns']].mean()).T
    df4 = pd.DataFrame(df[3][['AccX', 'AccY', 'AccZ', 'Acc-old', 'Acc-trns']].mean()).T

    dfmin1 = pd.DataFrame(df[0][['AccX', 'AccY', 'AccZ', 'Acc-old', 'Acc-trns']].min()).T
    dfmin2 = pd.DataFrame(df[1][['AccX', 'AccY', 'AccZ', 'Acc-old', 'Acc-trns']].min()).T
    dfmin3 = pd.DataFrame(df[2][['AccX', 'AccY', 'AccZ', 'Acc-old', 'Acc-trns']].min()).T
    dfmin4 = pd.DataFrame(df[3][['AccX', 'AccY', 'AccZ', 'Acc-old', 'Acc-trns']].min()).T

    dfmax1 = pd.DataFrame(df[0][['AccX', 'AccY', 'AccZ', 'Acc-old', 'Acc-trns']].max()).T
    dfmax2 = pd.DataFrame(df[1][['AccX', 'AccY', 'AccZ', 'Acc-old', 'Acc-trns']].max()).T
    dfmax3 = pd.DataFrame(df[2][['AccX', 'AccY', 'AccZ', 'Acc-old', 'Acc-trns']].max()).T
    dfmax4 = pd.DataFrame(df[3][['AccX', 'AccY', 'AccZ', 'Acc-old', 'Acc-trns']].max()).T

    MeanDF = df1.append(df2)
    MeanDF = MeanDF.append(df3)
    MeanDF = MeanDF.append(df4)

    MinDF = dfmin1.append(dfmin2)
    MinDF = MinDF.append(dfmin3)
    MinDF = MinDF.append(dfmin4)

    MaxDF = dfmax1.append(dfmax2)
    MaxDF = MaxDF.append(dfmax3)
    MaxDF = MaxDF.append(dfmax4)

    MeanDF.columns = ['mean1AccX', 'mean1AccY', 'mean1AccZ', 'mean1Acc-old', 'mean1Acc-turns']
    MinDF.columns = ['min1AccX', 'min1AccY', 'min1AccZ', 'min1Acc-old', 'min1Acc-turns']
    MaxDF.columns = ['max1AccX', 'max1AccY', 'max1AccZ', 'max1Acc-old', 'max1Acc-turns']

    MeanDF['min1AccX'] = MinDF['min1AccX']
    MeanDF['min1AccY'] = MinDF['min1AccY']
    MeanDF['min1AccZ'] = MinDF['min1AccZ']
    MeanDF['min1Acc-old'] = MinDF['min1Acc-old']
    MeanDF['min1Acc-turns'] = MinDF['min1Acc-turns']
    MeanDF['max1AccX'] = MaxDF['max1AccX']
    MeanDF['max1AccY'] = MaxDF['max1AccY']
    MeanDF['max1AccZ'] = MaxDF['max1AccZ']
    MeanDF['max1Acc-old'] = MaxDF['max1Acc-old']
    MeanDF['max1Acc-turns'] = MaxDF['max1Acc-turns']

    MeanDF = MeanDF.reset_index()
    MeanDF = MeanDF.drop(columns=['index'])

    return MeanDF


def makePredictions(models, standDF):
    rfModel, mlpModel, svcModel = models

    y_rf = rfModel.predict(standDF)
    y_mlp = mlpModel.predict(standDF)
    y_svc = svcModel.predict(standDF)

    standResults = pd.DataFrame()
    standResults['Random Forest'] = y_rf
    standResults['MLP'] = y_mlp
    standResults['SVC'] = y_svc
    print("Classifier-wise Predictions")
    print(standResults)

    y_rf_prob = pd.DataFrame(rfModel.predict_proba(standDF))
    y_rf_prob.columns = ['Standing', 'Walking', 'Running']
    print("\n\nRandom Forest Probability")
    print(y_rf_prob)

    y_mlp_prob = pd.DataFrame(mlpModel.predict_proba(standDF))
    y_mlp_prob.columns = ['Standing', 'Walking', 'Running']
    print("\n\nMLP Probability")
    print(y_mlp_prob)

    y_svc_prob = pd.DataFrame(svcModel.predict_proba(standDF))
    y_svc_prob.columns = ['Standing', 'Walking', 'Running']
    print("\n\nSVC Probability")
    print(y_svc_prob)


def main():
    standDF, walkDF, runDF = loadData()
    models = loadModels()

    standDF, walkDF, runDF = cleanData(standDF, walkDF, runDF)

    standDF = applyStatistics(standDF)
    walkDF = applyStatistics(walkDF)
    runDF = applyStatistics(runDF)

    print("\nPredictions for Standing [ 0 ]\n")
    makePredictions(models, standDF)

    print("\nPredictions for Walking [ 1 ]\n")
    makePredictions(models, walkDF)

    print("\nPredictions for Running [ 2 ]\n")
    makePredictions(models, runDF)


if __name__ == '__main__':
    main()
