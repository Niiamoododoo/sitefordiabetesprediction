from django.shortcuts import render


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
def index(request):
    return render(request, 'index.html')
def predict(request):
    return render(request, 'predict.html')
def Results(request):
    data = pd.read_csv(r"https://github.com/Niiamoododoo/sitefordiabetesprediction/blob/main/diabetesprediction/Diabetes_2_dataset.csv")

    X = data.drop("Outcome", axis=1)
    Y = data["Outcome"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, Y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    results1 = ""
    if pred == [1]:
        results1 = "Positive"
    else:
        results1 = "Negative"
    return render(request, 'predict.html', {"result2": results1})
