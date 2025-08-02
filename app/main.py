from fastapi import FastAPI
from app.routers import upload, cleaning, train, predict

app = FastAPI(title="Customer Churn Prediction API")

# Endpoints Configuring
app.include_router(upload.router)
app.include_router(cleaning.router)
app.include_router(train.router)
app.include_router(predict.router)

# Base endpoint to verify the app
@app.get("/")
def root():
    return {"Welcome to the Customer Churn Prediction API"}