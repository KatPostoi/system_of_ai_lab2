from task_2 import fit_model, load_data, plot_scatter, predict_gpa


def main():
    # 1. Загрузка данных
    X, y, data = load_data()

    print("Первые пять строк данных:")
    print(data.head())

    # 2–3. Построение графика и обучение модели
    plot_scatter(X, y)
    model = fit_model(X, y)

    # 4. Предсказание GPA по введённому SAT
    predict_gpa(model)


if __name__ == "__main__":
    main()
