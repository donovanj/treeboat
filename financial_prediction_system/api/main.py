from fastapi import FastAPI

app = FastAPI(
    title="Financial Prediction API",
    description="API for financial predictions using machine learning models",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Financial Prediction API"} 