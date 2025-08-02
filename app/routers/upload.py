from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
import pandas as pd
import os
from tempfile import NamedTemporaryFile

router = APIRouter(prefix="/upload", tags= ["Upload"])

@router.post("/")
async def upload_dataset(file: UploadFile, target_column: str = Form(...)):
    try:
        # save files temporarily
        temp = NamedTemporaryFile(delete=False)
        temp.write(await file.read())
        temp.close()

        #Load the uploaded file into Dataframe
        df = pd.read_csv(temp.name,encoding="utf-8")

        #Remove the temp file from the system
        os.remove(temp.name)

        #Check if the target variable is in dataset or not
        if target_column not in df.columns:
            return JSONResponse(
                status_code= 400,
                content={"error": f"Target column '{target_column}' not found in dataset"}
            )
        
        #Save the uploaded file in data folder for later accessing
        df.to_csv("data/raw_uploaded.csv", index=False)
        
        #Extract the MetaData
        column_type = df.dtypes.apply(lambda x:str(x)).to_dict()
        missing_values = df.isnull().sum().to_dict()
        sample_preview = df.head().to_dict(orient="records")

        

        return{
            "status" : "success",
            "target_column" : target_column,
            "columns" : list(df.columns),
            "column_types" : column_type,
            "missing_values" : missing_values,
            "sample_preview" : sample_preview
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error":str(e)}
        )