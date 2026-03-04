import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import numpy as np

def train_and_save_model():
    # Путь к датасету
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(project_root, 'data', 'bank_full.csv')

    if not os.path.exists(file_path):
        print(f"Ошибка: файл '{file_path}' не найден.")
        print("Проверьте, что датасет находится в папке data и называется bank_full.csv")
        return

    try:
        data = pd.read_csv(file_path, sep=';')
        if data.shape[1] <= 2 and ';' not in data.columns[0]:
            raise ValueError("Separator might be ';'. Trying ','.")
    except (pd.errors.ParserError, ValueError):
        try:
            data = pd.read_csv(file_path, sep=',')
            print("Файл загружен с разделителем ','.")
        except Exception as e:
            print(f"Ошибка при загрузке файла CSV с разделителем ',': {e}")
            return
    except Exception as e:
        print(f"Ошибка при загрузке файла CSV: {e}")
        return

    if 'y' not in data.columns:
        print("Ошибка: Колонка 'y' не найдена в датасете. Пожалуйста, убедитесь, что целевая переменная названа 'y'.")
        print(f"Доступные колонки: {data.columns.tolist()}")
        return

    # Признаки для модели
    selected_features = [
        'poutcome', 'contact', 'duration', 'housing', 'month',
        'previous', 'pdays', 'loan', 'age', 'day'
    ]

    missing_selected_features = [f for f in selected_features if f not in data.columns]
    if missing_selected_features:
        print(f"Ошибка: Признаки отсутствуют в датасете: {missing_selected_features}")
        print(f"Доступные колонки: {data.columns.tolist()}")
        return

    X = data[selected_features].copy()
    y = data['y']

    # Кодируем целевую переменную
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y)
    print(f"Целевая переменная 'y' закодирована: {y_encoder.classes_.tolist()} -> {list(range(len(y_encoder.classes_)))}")

    encoders = {}
    features_info = {}
    fallback_values = {}

    # Определяем типы признаков
    for column in X.columns:
        if X[column].dtype == 'object' or pd.api.types.is_string_dtype(X[column]):
            le = LabelEncoder()
            temp_col_for_mode = X[column].fillna(X[column].mode()[0])
            mode_value = temp_col_for_mode.mode()[0]

            le.fit(X[column].astype(str).unique())
            
            features_info[column] = {
                'type': 'categorical',
                'values': le.classes_.tolist(),
                'mode': mode_value
            }
            encoders[column] = le
            fallback_values[column] = mode_value

        else:
            try:
                X[column] = pd.to_numeric(X[column], errors='coerce')
            except Exception:
                pass
            features_info[column] = {'type': 'numerical', 'min': X[column].min(), 'max': X[column].max()}
            fallback_values[column] = X[column].median()

    # Применяем энкодеры и заполняем пропуски
    for column in X.columns:
        if X[column].dtype == 'object' or pd.api.types.is_string_dtype(X[column]):
            X[column] = X[column].fillna(fallback_values[column])
            X[column] = encoders[column].transform(X[column].astype(str))
        else:
            X[column] = X[column].fillna(fallback_values[column])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучаем XGBoost
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    top_10_features_names = selected_features
    print("\nМодель обучена только на следующих 10 признаках:")
    print(top_10_features_names)
    

    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Создана директория: '{models_dir}'")

    # Сохраняем параметры XGBoost
    try:
        xgboost_params = model.get_params()
        params_string = ""
        filtered_params = {k: v for k, v in xgboost_params.items() if v is not None}

        try:
            booster_params = model.get_xgb_params()
        except Exception:
            booster_params = {}

        merged = {**booster_params, **filtered_params}

        for key, value in merged.items():
            # Приводим numpy-типы к обычным Python-типам
            if isinstance(value, (np.integer, np.floating)):
                params_string += f"{key}: {value.item()}\n"
            else:
                params_string += f"{key}: {value}\n"

        params_filename = os.path.join(models_dir, 'xgboost_params.txt')
        with open(params_filename, 'w') as f:
            f.write(params_string)
        print(f"Параметры XGBoost успешно сохранены в '{params_filename}'")
    except Exception as e:
        print(f"Ошибка при сохранении параметров XGBoost в TXT: {e}")
    

    # Собираем артефакты
    all_artifacts = {
        'model': model,
        'features_list': X.columns.tolist(),
        'label_encoders': encoders,
        'features_info': features_info,
        'y_encoder': y_encoder,
        'fallback_values': fallback_values,
        'top_10_features_names': top_10_features_names
    }

    artifacts_filename = os.path.join(models_dir, 'all_bank_marketing_artifacts.joblib')
    joblib.dump(all_artifacts, artifacts_filename)
    print(f"\nВсе артефакты успешно обучены и сохранены в '{artifacts_filename}'")

if __name__ == '__main__':
    train_and_save_model()
