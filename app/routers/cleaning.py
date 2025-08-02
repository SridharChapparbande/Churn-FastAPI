from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import os
import joblib

router = APIRouter(prefix="/data_cleaning", tags=["Data Cleaning"])

#Shared file folders
DATA_FILE_PATH = "data/cleaned_data.csv" # Cleaned file path

@router.post("/")
def clean_data(
    missing_handling : str = Body(..., embed=True), # Input data
    encoding : str = Body(..., embed=True), # Input data
    scaling : str = Body(..., embed=True) # Input data
):
    try:
        df = pd.read_csv("data/raw_uploaded.csv")

        if "Churn" not in df.columns:
            return JSONResponse(status_code=400, content={"error": "Target column 'Churn' not found in dataset."})

        # Missing value handling
        if missing_handling == "drop":
            df = df.dropna()
        elif missing_handling == "impute":
            imputer = SimpleImputer(strategy="most_frequent")
            df[df.columns] = imputer.fit_transform(df)
        

        #Categorical columns handling
        if encoding == "label":
            label_encoders = {}
            for col in df.select_dtypes(include="object").columns:
                if col != "Churn":
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    label_encoders[col] = le
            joblib.dump(label_encoders, "models/label_encoders.pkl")

        elif encoding == "onehot":
            categorical_cols = [col for col in df.select_dtypes(include="object").columns if col != "Churn"]
            df = pd.get_dummies(df, columns=categorical_cols)


        # Handling scaling of numerical columns
        features = df.drop(columns=["Churn"])
        if scaling == "standard":
            scaler = StandardScaler()
        elif scaling == "minmax":
            scaler = MinMaxScaler()

        scaled_features = scaler.fit_transform(features)
        df_scaled = pd.DataFrame(scaled_features, columns=features.columns)

        # Add target column back
        df_scaled["Churn"] = df["Churn"].values[:df_scaled.shape[0]]

        joblib.dump(scaler, "models/scaler.pkl")

        df_scaled.to_csv(DATA_FILE_PATH, index=False)

        return{
            "status" : "success",
            "shape" : df_scaled.shape,
            "columns" : list(df_scaled.columns),
            "preview" :df_scaled.head().to_dict(orient="records")
        }
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error":str(e)})

