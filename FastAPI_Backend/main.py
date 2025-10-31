from fastapi import FastAPI
from pydantic import BaseModel, conlist
from typing import List, Optional
import pandas as pd
from model import recommend, output_recommended_recipes
import uvicorn

dataset = pd.read_csv('../Data/dataset.csv', compression='gzip')

app = FastAPI()

class Params(BaseModel):
    n_neighbors: int = 5
    return_distance: bool = False

class PredictionIn(BaseModel):
    # v2: dùng min_length/max_length (không phải min_items/max_items)
    nutrition_input: conlist(float, min_length=9, max_length=9)
    ingredients: List[str] = []
    params: Optional[Params] = None

class Recipe(BaseModel):
    Name: str
    CookTime: int
    PrepTime: int
    TotalTime: int
    RecipeIngredientParts: List[str]
    Calories: float
    FatContent: float
    SaturatedFatContent: float
    CholesterolContent: float
    SodiumContent: float
    CarbohydrateContent: float
    FiberContent: float
    SugarContent: float
    ProteinContent: float
    RecipeInstructions: List[str]

class PredictionOut(BaseModel):
    output: Optional[List[Recipe]] = None

@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/predict/", response_model=PredictionOut)
def predict(prediction_input: PredictionIn):
    # v2: .model_dump() thay cho .dict()
    p = prediction_input.params.model_dump() if prediction_input.params else {}
    df = recommend(dataset, prediction_input.nutrition_input, prediction_input.ingredients, p)
    output = output_recommended_recipes(df)
    return {"output": output if output is not None else None}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# http://localhost:8000