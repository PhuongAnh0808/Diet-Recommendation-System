import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def scaling(dataframe):
    scaler=StandardScaler()
    prep_data=scaler.fit_transform(dataframe.iloc[:,6:15].to_numpy())
    return prep_data,scaler

def nn_predictor(prep_data):
    neigh = NearestNeighbors(metric='cosine',algorithm='brute')
    neigh.fit(prep_data)
    return neigh

def build_pipeline(neigh,scaler,params):
    transformer = FunctionTransformer(neigh.kneighbors,kw_args=params)
    pipeline=Pipeline([('std_scaler',scaler),('NN',transformer)])
    return pipeline

def extract_data(dataframe,ingredients):
    extracted_data=dataframe.copy()
    extracted_data=extract_ingredient_filtered_data(extracted_data,ingredients)
    return extracted_data
    
def extract_ingredient_filtered_data(dataframe,ingredients):
    extracted_data=dataframe.copy()
    regex_string=''.join(map(lambda x:f'(?=.*{x})',ingredients))
    extracted_data=extracted_data[extracted_data['RecipeIngredientParts'].str.contains(regex_string,regex=True,flags=re.IGNORECASE)]
    return extracted_data

def apply_pipeline(pipeline,_input,extracted_data):
    _input=np.array(_input).reshape(1,-1)
    return extracted_data.iloc[pipeline.transform(_input)[0]]

def recommend(dataframe,_input,ingredients=[],params={'n_neighbors':5,'return_distance':False}):
        extracted_data=extract_data(dataframe,ingredients)
        if extracted_data.shape[0]>=params['n_neighbors']:
            prep_data,scaler=scaling(extracted_data)
            neigh=nn_predictor(prep_data)
            pipeline=build_pipeline(neigh,scaler,params)
            return apply_pipeline(pipeline,_input,extracted_data)
        else:
            return None

def extract_quoted_strings(s):
    # Find all the strings inside double quotes
    strings = re.findall(r'"([^"]*)"', s)
    # Join the strings with 'and'
    return strings

def output_recommended_recipes(dataframe):
    if dataframe is not None:
        output=dataframe.copy()
        output=output.to_dict("records")
        for recipe in output:
            recipe['RecipeIngredientParts']=extract_quoted_strings(recipe['RecipeIngredientParts'])
            recipe['RecipeInstructions']=extract_quoted_strings(recipe['RecipeInstructions'])
    else:
        output=None
    return output

# ======================== MODEL EVALUATION ========================

from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans


def evaluate_model(df, k_list=[3, 5, 7, 10]):
    # Select needed nutrition columns
    nutri_cols = ["Calories", "FatContent", "SaturatedFatContent", "CholesterolContent",
                  "SodiumContent", "CarbohydrateContent", "FiberContent", "SugarContent", "ProteinContent"]

    # Convert nutrition columns to numeric
    df[nutri_cols] = df[nutri_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=nutri_cols)

    # Sample for evaluation
    df_eval = df.sample(n=min(1200, len(df)), random_state=42).reset_index(drop=True)
    nutri_features = df_eval[nutri_cols].to_numpy()

    scaler = StandardScaler()
    nutri_scaled = scaler.fit_transform(nutri_features)

    # Silhouette Score
    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=256, random_state=42)
    labels = kmeans.fit_predict(nutri_scaled)
    sil_score = round(silhouette_score(nutri_scaled, labels), 4)

    results = {"silhouette_score": sil_score, "avg_distance_per_k": {}}

    # Nearest Neighbor distance evaluation
    neigh = NearestNeighbors(metric="cosine", algorithm="brute")
    neigh.fit(nutri_scaled)

    for k in k_list:
        distances, _ = neigh.kneighbors(nutri_scaled, n_neighbors=k)
        avg_dist = distances.mean()
        results["avg_distance_per_k"][k] = round(avg_dist, 4)

    return results


# ======================== RUN FOR TEST ========================

if __name__ == "__main__":
    import pandas as pd

    print("\nLoading dataset...")
    df = pd.read_csv("../Data/dataset.csv", compression='gzip', low_memory=False)

    print("\nEvaluating model...")
    eval_results = evaluate_model(df)

    print("\n===== MODEL EVALUATION RESULTS =====")
    print(f"Silhouette Score: {eval_results['silhouette_score']:.4f}")

    print("\nAverage Distance per k:")
    for k, dist in eval_results["avg_distance_per_k"].items():
        print(f"  k = {k:<3} → average cosine distance = {dist:.4f}")

