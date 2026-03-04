from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import uvicorn
import numpy as np
import xgboost as xgb
from endpoint import BankMarketingFeatures

app = FastAPI(title="Bank Marketing Prediction API")

# Артефакты модели
model = None
model_features = None
label_encoders = None
features_info = None
y_encoder = None
fallback_values = None
top_10_features_names = None


# Загружаем артефакты при запуске
@app.on_event("startup")
async def load_artifacts():
    global model, model_features, label_encoders, features_info, y_encoder, fallback_values, top_10_features_names
    try:
        models_dir = 'models'
        artifacts_path = os.path.join(models_dir, 'all_bank_marketing_artifacts.joblib')
        params_txt_path = os.path.join(models_dir, 'xgboost_params.txt')

        if not os.path.exists(artifacts_path):
            print(f"Файл артефактов не найден: '{artifacts_path}'")
            raise FileNotFoundError(f"Артефакты не найдены: {artifacts_path}")
        
        if not os.path.exists(params_txt_path):
            print(f"Файл параметров XGBoost не найден: '{params_txt_path}'")
            raise FileNotFoundError(f"Параметры XGBoost не найдены: {params_txt_path}")


        all_artifacts = joblib.load(artifacts_path)
        print(f"Артефакты загружены: '{artifacts_path}'")

        model = all_artifacts['model']
        model_features = all_artifacts['features_list']
        label_encoders = all_artifacts['label_encoders']
        features_info = all_artifacts['features_info']
        y_encoder = all_artifacts['y_encoder']
        fallback_values = all_artifacts['fallback_values']
        top_10_features_names = all_artifacts.get('top_10_features_names', [])

        print("Модель и вспомогательные данные загружены")
    except Exception as e:
        print(f"Ошибка загрузки артефактов: {e}")
        raise HTTPException(status_code=500, detail=f"Не удалось загрузить файлы модели: {e}")

@app.get("/", summary="Корневой эндпоинт")
async def read_root():
    return {"message": "Добро пожаловать в Bank Marketing Prediction API! Наберите /docs для получения дополнительной информации."}

@app.get("/features_info", summary="Получить информацию о признаках для построения UI")
async def get_features_info():
    if features_info is None:
        raise HTTPException(status_code=500, detail="Информация о признаках пока не загружена")
    return features_info

@app.get("/xgboost_params", summary="Получить параметры загруженной модели XGBoost (txt формат)", response_class=PlainTextResponse)
async def get_xgboost_params():
    # Возвращаем параметры из txt
    params_txt_path = os.path.join('models', 'xgboost_params.txt')
    if not os.path.exists(params_txt_path):
        raise HTTPException(status_code=500, detail=f"Файл параметров не найден: {params_txt_path}")
    
    try:
        with open(params_txt_path, 'r') as f:
            params_content = f.read()
        return params_content
    except Exception as e:
        print(f"Ошибка чтения файла параметров: {e}")
        raise HTTPException(status_code=500, detail=f"Не удалось прочитать файл параметров: {e}")


@app.post("/predict", summary="Предсказать результат банковского маркетинга")
async def predict(features: BankMarketingFeatures):
    # Делаем предсказание по входным данным
    if model is None or model_features is None or label_encoders is None or features_info is None:
        raise HTTPException(status_code=500, detail="Модель еще не загружена")

    input_dict = features.dict()
    
    input_df = pd.DataFrame([input_dict])

    # Кодируем категориальные поля
    for column, encoder in label_encoders.items():
        if column in input_df.columns:
            try:
                if input_df[column].iloc[0] not in encoder.classes_:
                    raise HTTPException(status_code=400, detail=f"Неизвестное значение '{input_df[column].iloc[0]}' для поля '{column}'. Возможные значения: {encoder.classes_.tolist()}")
                
                input_df[column] = encoder.transform(input_df[column])
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Ошибка кодирования поля '{column}': {e}")
        else:
            raise HTTPException(status_code=400, detail=f"В запросе нет обязательного поля: {column}")

    # Выставляем порядок колонок как в обучении
    try:
        input_df = input_df[model_features]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Не хватает поля в запросе: {e}")

    prediction_raw = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0].tolist()

    prediction_text = y_encoder.inverse_transform([prediction_raw])[0]

    return {
        "prediction_raw": int(prediction_raw),
        "prediction_text": prediction_text,
        "probability_class_0": probability[0],
        "probability_class_1": probability[1]
    }

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
