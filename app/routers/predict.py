from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
import os
from app.utils.genai_utils import generate_explanation

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("/")
async def predict_and_explain(file : UploadFile, model_name: str = Form(...)):
    try:
        #Load the model first
        model_path = f"models/{model_name}.pkl"
        if not os.path.exists(model_path):
            return JSONResponse(status_code=500, content= {"error": f"Model '{model_name}' not found"})
        model = joblib.load(model_path)


        # Loading the dataset
        df_test = pd.read_csv(file.file)

        #Load and apply scaling and label encoding
        encoder_path = "models/label_encoders.pkl"
        scaler_path = "models/scaler.pkl"

        if "Churn" in df_test.columns:
            df_test = df_test.drop(columns=["Churn"])

        if os.path.exists(encoder_path):
            label_encoders = joblib.load(encoder_path)
            for col in df_test.select_dtypes(include="object").columns:
                if col in label_encoders:
                    df_test[col] = label_encoders[col].transform(df_test[col])

        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            df_test[df_test.columns] = scaler.transform(df_test[df_test.columns ])

        #Predict
        predictions = model.predict(df_test)

        #Store predictions
        df_test["prediction"] = predictions

        #Generate explanation using GenAI
        explanations = []
        for idx, row in df_test.iterrows():
            explanation = generate_explanation(row.to_dict())
            explanations.append(explanation)
        
        df_test["explanation"] = explanations

        return{
            "status" : "success",
            "results" : df_test.head(10).to_dict(orient="records")
        }
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error":str(e)})
