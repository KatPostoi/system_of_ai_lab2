import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def load_data():
    """Загрузка данных SAT и GPA."""
    url = "https://raw.githubusercontent.com/tehmeerali786/360-Data-Science-/master/1.01.%20Simple%20linear%20regression.csv"
    try:
        data = pd.read_csv(url)
    except:
        data = pd.read_csv("1.01. Simple linear regression.csv")  # если файл скачан вручную
    X = data[["SAT"]]
    y = data["GPA"]
    return X, y, data


def plot_scatter(X, y):
    """Построение диаграммы рассеяния и подписей осей."""
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color="blue", label="Реальные данные (SAT → GPA)")
    plt.xlabel("SAT (академический тест)")
    plt.ylabel("GPA (средний балл аттестата)")
    plt.title("Зависимость GPA от SAT")



def fit_model(X, y):
    """Обучение линейной регрессии и возврат модели."""
    model = LinearRegression()
    model.fit(X, y)

    # прямая линии регрессии
    X_min = X.values.min()
    X_max = X.values.max()
    X_line = np.linspace(X_min, X_max, 100).reshape(-1, 1)
    y_line = model.predict(X_line)

    plt.plot(X_line, y_line, color="red", linewidth=2, label="Линия регрессии")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return model



def predict_gpa(model):
    """Интерфейс ввода SAT и вывод предсказанного GPA."""
    while True:
        try:
            sat_input = float(input("Введите значение SAT (например, 1700) или 0 для выхода: "))
            if sat_input == 0:
                break
            if sat_input < 400 or sat_input > 2400:
                print("Значение SAT должно быть в диапазоне 400–2400.")
                continue
            gpa_pred = model.predict([[sat_input]])[0]
            print(f"Предсказанный GPA для SAT = {sat_input}: {gpa_pred:.3f}\n")
        except ValueError:
            print("Введите число.")
