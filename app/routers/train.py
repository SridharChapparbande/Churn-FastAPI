from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import os
# Models import
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
#Metrics import
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

router = APIRouter(prefix="/train_model", tags=["Model Training"])

@router.post("/")
def train_models(models : list = Body(...,embed=True)):
    try:
        # loading the cleaned data from previous step
        df = pd.read_csv("data/cleaned_data.csv")

        if "Churn" not in df.columns:
            return JSONResponse(status_code=500, content={"error":"Target column not found in dataset"})
        
        X = df.drop(columns=["Churn"])
        y = df["Churn"]

        #Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 10)
        results = {}

        for model_name in models:
            if model_name == "LogisticRegression":
                model = LogisticRegression(max_iter=1000)
            elif model_name == "RandomForest":
                model = RandomForestClassifier()
            elif model_name == "XGBoost":
                model = XGBClassifier(use_label_encoder = False, eval_metric = "logloss")
            else:
                continue

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Save the model for inferencing
            model_path = f"models/{model_name}.pkl"
            joblib.dump(model, model_path)

            #Final evaluation 
            results[model_name] = {
                "accuracy" : round(accuracy_score(y_test, y_pred),4),
                "precision" : round(precision_score(y_test, y_pred, average= 'weighted'),4),
                "recall" : round(recall_score(y_test, y_pred, average= 'weighted'),4),
                "f1_score" : round(f1_score(y_test, y_pred, average= 'weighted'),4),
                "confusion_matrix" : confusion_matrix(y_test,y_pred).tolist()          
            }

            return{
                "status" : "success",
                "models_trained" : list(results.keys()),
                "evaluation" : results
            }
    
    except Exception as e:
        return JSONResponse(status_code=500, content= {"error":str(e)})
            