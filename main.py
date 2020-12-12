from joblib import load
import sys
import numpy as np
import pandas as pd
from scipy import signal


def loadData(infile):
    DF = []

    for i in range(1):
        DF.append(pd.read_csv(infile))
    return DF


def loadModels():
    # # Model Load
    rfModel = load('models/random_forest.joblib')
    mlpModel = load('models/neural_network.joblib')
    svcModel = load('models/svc_classifier.joblib')

    return rfModel, mlpModel, svcModel


def cleanData(DF):
    # using low pass filter
    b, a = signal.butter(3, 0.1, btype='lowpass', analog=False)

    data_np = list(DF)

    butter_worth_data = []

    for i in range(1):
        butter_worth_dataX = signal.filtfilt(b, a, data_np[i]["gFx"])
        butter_worth_dataY = signal.filtfilt(b, a, data_np[i]["gFy"])
        butter_worth_dataZ = signal.filtfilt(b, a, data_np[i]["gFz"])
        butter_worth_data.append((pd.DataFrame(list(
            zip(data_np[i]['time'], butter_worth_dataX, butter_worth_dataY, butter_worth_dataZ,
                data_np[i]['acc'])),
            columns=["Time", "AccX", "AccY", "AccZ", "Acc-old"])))

    for i in range(1):
        butter_worth_data[i]["Acc-trns"] = np.sqrt(butter_worth_data[i]["AccX"] ** 2 *
                                                   butter_worth_data[i]["AccY"] ** 2 *
                                                   butter_worth_data[i]["AccZ"] ** 2)

    # Update current data with cleaned data
    newRun = []

    for i in range(1):
        newRun.append(butter_worth_data[i])

    return newRun


def applyStatistics(df):
    # ### Finding mean, min and max

    df1 = pd.DataFrame(df[0][['AccX', 'AccY', 'AccZ', 'Acc-old', 'Acc-trns']].mean()).T
    dfmin = pd.DataFrame(df[0][['AccX', 'AccY', 'AccZ', 'Acc-old', 'Acc-trns']].min()).T
    dfmax = pd.DataFrame(df[0][['AccX', 'AccY', 'AccZ', 'Acc-old', 'Acc-trns']].max()).T

    MeanDF = df1
    MinDF = dfmin
    MaxDF = dfmax

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


def makePredictions(models, DF):
    rfModel, mlpModel, svcModel = models

    y_rf = rfModel.predict(DF)
    y_mlp = mlpModel.predict(DF)
    y_svc = svcModel.predict(DF)

    standResults = pd.DataFrame()
    standResults['Random Forest'] = y_rf
    standResults['MLP'] = y_mlp
    standResults['SVC'] = y_svc
    print("Classifier-wise Predictions")
    print(standResults)

    y_rf_prob = pd.DataFrame(rfModel.predict_proba(DF))
    y_rf_prob.columns = ['Standing', 'Walking', 'Running']
    print("\n\nRandom Forest Probability")
    print(y_rf_prob)

    y_mlp_prob = pd.DataFrame(mlpModel.predict_proba(DF))
    y_mlp_prob.columns = ['Standing', 'Walking', 'Running']
    print("\n\nMLP Probability")
    print(y_mlp_prob)

    y_svc_prob = pd.DataFrame(svcModel.predict_proba(DF))
    y_svc_prob.columns = ['Standing', 'Walking', 'Running']
    print("\n\nSVC Probability")
    print(y_svc_prob)


def main(infile):
    DF = loadData(infile)
    models = loadModels()

    DF = cleanData(DF)

    DF = applyStatistics(DF)

    print("\nPredictions for Dataset:\n\t0: Standing\n\t1: Walking\n\t2: Running\n")
    makePredictions(models, DF)


if __name__ == '__main__':
    infile = sys.argv[1]
    main(infile)
