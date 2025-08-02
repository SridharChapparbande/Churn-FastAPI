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

        if "target" in df.columns:
            target = df["target"]
            df = df.drop(columns=["target"])
        else:
            target = None
            
        # Missing value handling
        if missing_handling == "drop":
            df = df.dropna()
        elif missing_handling == "impute":
            imputer = SimpleImputer(strategy="most_frequent")
            df[df.columns] = imputer.fit_transform(df)
        

        #Categorical columns handling
        if encoding == "label":
            label_encoder = {}
            for col in df.select_dtypes(include="object").columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoder[col] = le
            joblib.dump(label_encoder, "models/label_encoders.pkl")

        elif encoding == "onehot":
            df = pd.get_dummies(df)

        # Handling scaling of numerical columns
        if scaling == "standard":
            scaler = StandardScaler()
        elif scaling == "minmax":
            scaler = MinMaxScaler()
        
        df[df.columns] = scaler.fit_transform(df[df.columns])
        joblib.dump(scaler, "models/scaler.pkl")

        # Adding the target variable back to the dataset
        if target is not None:
            df['target'] = target.values[:df.shape[0]]

        # Saving the cleaned data
        df.to_csv(DATA_FILE_PATH, index= False)

        return{
            "status" : "success",
            "shape" : df.shape,
            "columns" : list(df.columns),
            "preview" :df.head().to_dict(orient="records")
        }
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error":str(e)})

