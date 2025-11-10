import azure.functions as func
import logging
import pandas as pd
import base64
from io import BytesIO
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from azure.storage.blob import BlobServiceClient
from sklearn.metrics import silhouette_score
import datetime


app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="convertexceltojson", auth_level=func.AuthLevel.FUNCTION)
def convertexceltojson(req: func.HttpRequest) -> func.HttpResponse:
    try:
        file_bytes = None

        # Try JSON first
        try:
            body = req.get_json()
            if "$content" in body:
                logging.info("Detected JSON input with $content")
                file_bytes = base64.b64decode(body["$content"])
            else:
                logging.warning("JSON body found but missing '$content'")
                return func.HttpResponse(
                    "Missing '$content' in request body",
                    status_code=400
                )
        except ValueError:
            # If JSON parsing fails, fall back to raw binary
            logging.info("Falling back to binary input")
            file_bytes = req.get_body()

        if not file_bytes:
            return func.HttpResponse("No file data found", status_code=400)

        # Debug log file size
        logging.info(f"Received file size: {len(file_bytes)} bytes")

        # Read Excel into DataFrame
        df = pd.read_excel(BytesIO(file_bytes), engine="openpyxl")

        # Debug info
        logging.info(f"DataFrame shape: {df.shape}")

        result_json = df.to_json(orient="records")

        return func.HttpResponse(
            result_json,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Error converting Excel to JSON: {e}")
        return func.HttpResponse(str(e), status_code=500)
    
model = None

def get_model():
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def load_datasets_from_local(base_path="dataset"):
    """Loads all required Excel datasets into a global dictionary."""
    global DATASETS
    if DATASETS:
        return DATASETS
    
    dataset_files = {
        "df_ureq": "Usecase Requirement.xlsx",
        "df_talent": "(Pseudonym) Talent Data.xlsx",
        "df_skillinv": "(Pseudonym) Skill Inventory.xlsx",
        "df_hist": "(Pseudonym) History Usecase.xlsx",
        "df_eval": "(Pseudonym) Evaluation Scores.xlsx",
        "df_assign": "(Pseudonym) Assignment Data.xlsx"
    }
    
    loaded_datasets = {}
    try:
        for key, filename in dataset_files.items():
            file_path = os.path.join(base_path, filename)
            if os.path.exists(file_path):
                loaded_datasets[key] = pd.read_excel(file_path)
                logging.info(f"Successfully loaded {filename}")
            else:
                raise FileNotFoundError(f"Dataset file not found at {file_path}")
        DATASETS = loaded_datasets
        return DATASETS
    except Exception as e:
        logging.error(f"Error loading datasets: {e}")
        return None

def load_datasets_from_json(json_data):
    """Loads datasets from JSON data received in the request body."""
    global DATASETS
    
    DATASETS = {}

    try:
        loaded_datasets = {}
        
        # Map of expected filenames to dataset keys
        filename_mapping = {
            "Usecase Requirement.xlsx": "df_ureq",
            "(Pseudonym) Talent Data.xlsx": "df_talent",
            "(Pseudonym) Skill Inventory.xlsx": "df_skillinv",
            "(Pseudonym) History Usecase.xlsx": "df_hist",
            "(Pseudonym) Evaluation Scores.xlsx": "df_eval",
            "(Pseudonym) Assignment Data.xlsx": "df_assign"
        }
        
        # Process each file in the JSON array
        for file_obj in json_data:
            filename = file_obj.get("FileName")
            data = file_obj.get("Data")
            
            if not filename or data is None:
                logging.warning(f"Skipping invalid entry: {file_obj}")
                continue
            
            # Find the corresponding dataset key
            dataset_key = filename_mapping.get(filename)
            
            if dataset_key:
                # Convert the data array to a pandas DataFrame
                df = pd.DataFrame(data)
                loaded_datasets[dataset_key] = df
                logging.info(f"Successfully loaded {filename} with {len(df)} rows")
            else:
                logging.warning(f"Unknown filename: {filename}")
        
        # Check if all required datasets were loaded
        required_keys = set(filename_mapping.values())
        loaded_keys = set(loaded_datasets.keys())
        missing_keys = required_keys - loaded_keys
        
        if missing_keys:
            logging.warning(f"Missing datasets: {missing_keys}")
        
        DATASETS = loaded_datasets
        return DATASETS
        
    except Exception as e:
        logging.error(f"Error loading datasets from JSON: {e}")
        return None

# --- HELPER & PREPROCESSING FUNCTIONS ---
def convert_to_months(duration):
    years, months = 0, 0
    parts = str(duration).split()
    for part in parts:
        if 'tahun' in part:
            try: years = int(parts[parts.index(part)-1])
            except (ValueError, IndexError): years = 0
        elif 'bulan' in part:
            try: months = int(parts[parts.index(part)-1])
            except (ValueError, IndexError): months = 0
    return years * 12 + months

def minmax_scaling(series):
    if series.max() == series.min(): return 0.5
    return (series - series.min()) / (series.max() - series.min())

# --- CORE LOGIC FUNCTIONS ---
def find_talent_for_use_case(df_ureq, df_skillinv, df_talent, df_eval, df_hist, df_assign):
    """Core logic to find and score talent based on all available use case requirements."""
    sentence_model = get_model()
    skillsets = df_skillinv.columns[2:-1].tolist()
    # The agg_sentences creation is now done in the main function to be available for query matching
    if 'agg_sentences' not in df_ureq.columns:
        df_ureq['agg_sentences'] = df_ureq['Responsibilities'] + " " + df_ureq['Skill 1'].fillna('') + " " + df_ureq['Skill 2'].fillna('')
    
    results = []
    for _, row in df_ureq.iterrows():
        corpus = [row['agg_sentences']] + skillsets
        embeddings = sentence_model.encode(corpus)
        embeddings = np.array(embeddings)
        sim_scores = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1:])[0]
        for i, skill in enumerate(skillsets):
            results.append([row['Responsibilities'], row['Skill 1'], row['Skill 2'], row['Role'], skill, sim_scores[i]])
    
    df_results = pd.DataFrame(results, columns=['Responsibilities', 'Skill 1', 'Skill 2', 'Role', 'Skillset', 'Similarity score'])
    df_results_filtered = df_results[df_results['Similarity score'] >= 0.3] # Use a base threshold

    unique_ids_roles = df_skillinv[['UNIQUE ID', 'Role']].drop_duplicates().rename(columns={'Role': 'Role Person'})
    merged_df = df_results_filtered.merge(unique_ids_roles, how='cross')

    def get_skill_score(row):
        try:
            score = df_skillinv.loc[df_skillinv['UNIQUE ID'] == row['UNIQUE ID'], row['Skillset']]
            return float(score.values[0]) if not score.empty else np.nan
        except Exception:
            return np.nan

    merged_df['Skill Score'] = merged_df.apply(get_skill_score, axis=1)
    df_search = merged_df.groupby(['Responsibilities', 'Role', 'UNIQUE ID', 'Role Person'], as_index=False).agg(Avg_SkillScore=('Skill Score', 'mean'))

    df_numerical = df_skillinv.select_dtypes(include=[np.number]).dropna(axis=1)
    if len(df_numerical) >= 4 and len(df_numerical.columns) >= 2:
        pca = PCA(n_components=2)
        df_pca = pca.fit_transform(df_numerical)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
        df_skillinv['Cluster'] = kmeans.fit_predict(df_pca)
        df_search = df_search.merge(df_skillinv[['UNIQUE ID', 'Cluster']], on='UNIQUE ID', how='left')
    else:
        df_search['Cluster'] = -1 # Default cluster if not possible

    df_talent['Durasi Bulan'] = df_talent['LAMA KERJA BERJALAN'].apply(convert_to_months)
    df_agg_talent = pd.merge(df_talent, df_eval, on='UNIQUE ID', how='inner')
    df_agg_talent['scoring_eval'] = df_agg_talent['Capability Score']
    df_merged = pd.merge(df_search, df_agg_talent, on='UNIQUE ID', how='inner')

    df_hist_count = df_hist.groupby("UNIQUE ID")["PRODUCT / USECASE"].nunique().reset_index(name="job_count")
    df_final = pd.merge(df_merged, df_hist_count, on='UNIQUE ID', how='left').fillna({'job_count': 0})

    a, b, r = 0.47, 0.53, 1.2
    df_final['d'] = (df_final['Avg_SkillScore'] * a + df_final['scoring_eval'] * b)
    df_final['finalscore'] = df_final.apply(lambda row: row['d'] * r if row['Role Person'] == row['Role'] else row['d'], axis=1)
    df_final['finalscore_scaled'] = df_final.groupby('Responsibilities')['finalscore'].transform(minmax_scaling)
    
    return df_final

@app.route(route="talent_recommender", auth_level=func.AuthLevel.FUNCTION)
def talent_recommender(req: func.HttpRequest) -> func.HttpResponse:
    """
    Modified version — runs full background analysis (like final_df) for all jobs and all talents.
    Does not require a 'query' or 'threshold' parameter.
    """

    logging.info('Talent Recommender (Full Analysis) started.')

    try:
        # Parse request body
        try:
            req_body = req.get_json() if req.get_body() else {}
        except ValueError:
            return func.HttpResponse(
                json.dumps({"error": "Invalid JSON in request body"}),
                status_code=400
            )

        # Load datasets from request
        datasets_json = req_body.get('datasets')
        datasets = load_datasets_from_json(datasets_json)
        if not datasets:
            return func.HttpResponse(
                json.dumps({"error": "Could not load datasets. Check logs."}),
                status_code=500
            )

        # --- Run full pipeline (same as run_background_analysis) ---
        df_final = find_talent_for_use_case(
            df_ureq=datasets["df_ureq"],
            df_skillinv=datasets["df_skillinv"],
            df_talent=datasets["df_talent"],
            df_eval=datasets["df_eval"],
            df_hist=datasets["df_hist"],
            df_assign=datasets["df_assign"]
        )

        # Optional: reorder or limit columns (to mirror your previous final_df)
        columns_order = [
            "Responsibilities", "Skill 1", "Skill 2", "Role", "agg_sentences",
            "UNIQUE ID", "Role Person", "Skillset", "Avg_SkillScore", "Cluster",
            "ROLE", "LAMA KERJA BERJALAN", "GRADE", "Durasi Bulan",
            "Technical Score (29,06%)", "Personal Evaluation Score (49,17%)",
            "Discipline Score (15,06%)", "Development Score (7%)",
            "Expert Judgement", "Capability Score", "scoring_eval",
            "job_count", "d", "finalscore", "finalscore_scaled"
        ]
        existing_cols = [c for c in columns_order if c in df_final.columns]
        df_final = df_final[existing_cols]

        # --- Return as JSON ---
        result_json = df_final.to_json(orient="records")
        return func.HttpResponse(
            result_json,
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error in talent_recommender full analysis: {e}", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500
        )
        
@app.route(route="upload_results_to_blob", auth_level=func.AuthLevel.FUNCTION)
def upload_results_to_blob(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # Parse JSON from request
        body = req.get_json()

        # Ensure data is a list
        if isinstance(body, dict):
            data_list = [body]
        elif isinstance(body, list):
            data_list = body
        else:
            return func.HttpResponse("Invalid JSON format", status_code=400)

        # Convert to DataFrame for easier deduplication
        df = pd.DataFrame(data_list)

        # Drop duplicates based on relevant columns
        if not df.empty:
            df = df.drop_duplicates(subset=["UNIQUE ID", "Responsibilities"], keep="first")

        # Limit to first 15 entries (for testing)

        # Convert back to JSON
        clean_json = df.to_json(orient="records")

        # --- Upload to Blob Storage ---
        blob_conn_str = os.environ["AzureWebJobsStorage"]
        blob_service = BlobServiceClient.from_connection_string(blob_conn_str)
        container_name = "blobcleancontainer"
        container = blob_service.get_container_client(container_name)

        try:
            container.create_container()
        except Exception as e:
            if "ContainerAlreadyExists" not in str(e):
                raise

        # Save timestamped copy
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        blob_name = f"talent_results_{timestamp}.json"
        blob_client = container.get_blob_client(blob_name)
        blob_client.upload_blob(clean_json, overwrite=True)

        # Save or overwrite "latest.json"
        latest_blob = container.get_blob_client("latest.json")
        latest_blob.upload_blob(clean_json, overwrite=True)

        return func.HttpResponse(
            json.dumps({
                "message": "Upload successful, duplicates removed.",
                "entries_uploaded": len(df),
                "file": blob_name
            }),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        return func.HttpResponse(str(e), status_code=500)


@app.blob_trigger(arg_name="myblob", ource = "EventGrid", path="mycontainer",
                               connection="4a07e8_STORAGE") 
def EventGridBlobTrigger(myblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob"
                f"Name: {myblob.name}"
                f"Blob Size: {myblob.length} bytes")


# This example uses SDK types to directly access the underlying BlobClient object provided by the Blob storage trigger.
# To use, uncomment the section below and add azurefunctions-extensions-bindings-blob to your requirements.txt file
# Ref: aka.ms/functions-sdk-blob-python
#
# import azurefunctions.extensions.bindings.blob as blob
# @app.blob_trigger(arg_name="client", path="mycontainer",
#                   connection="4a07e8_STORAGE")
# def EventGridBlobTrigger(client: blob.BlobClient):
#     logging.info(
#         f"Python blob trigger function processed blob \n"
#         f"Properties: {client.get_blob_properties()}\n"
#         f"Blob content head: {client.download_blob().read(size=1)}"
#     )
