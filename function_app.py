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
    
def load_datasets_from_json_newfile(json_data):
    """Loads datasets from JSON data received in the request body (new Self-Assessment version)."""
    global DATASETS
    DATASETS = {}

    try:
        loaded_datasets = {}
        
        # Updated mapping: Self-Assessment replaces Skill Inventory
        filename_mapping = {
            "Usecase Requirement.xlsx": "df_ureq",
            "(Pseudonym) Talent Data.xlsx": "df_talent",
            "(Pseudonym) Self-Assessment Score.xlsx": "df_selfassessment",
            "(Pseudonym) History Usecase.xlsx": "df_hist",
            "(Pseudonym) Capability Scores.xlsx": "df_eval",
            "(Pseudonym) Assignment Data.xlsx": "df_assign"
        }
        
        for file_obj in json_data:
            filename = file_obj.get("FileName")
            data = file_obj.get("Data")
            if not filename or data is None:
                logging.warning(f"Skipping invalid entry: {file_obj}")
                continue

            dataset_key = filename_mapping.get(filename)
            if dataset_key:
                df = pd.DataFrame(data)
                loaded_datasets[dataset_key] = df
                logging.info(f"Successfully loaded {filename} with {len(df)} rows")
            else:
                logging.warning(f"Unknown filename: {filename}")
        
        DATASETS = loaded_datasets
        return DATASETS
    except Exception as e:
        logging.error(f"Error loading datasets from JSON: {e}")
        return None
    
def is_json_valid(df):
    """ Row-level validation for latest.json """
    if df.empty:
        return False
    
    required_columns = ["UNIQUE ID", "Responsibilities"]

    # FAIL if ANY ROW is missing ANY required field
    for col in required_columns:
        if col not in df.columns:
            return False
        if df[col].isnull().any():      # <-- THIS IS WHAT YOU WERE MISSING
            return False

    return True


def restore_previous_version(container):
    """ Restore previous timestamped version as latest.json """
    blobs = list(container.list_blobs(name_starts_with="talent_results_"))

    if not blobs:
        return None  # nothing to restore

    # Sort by timestamp in filename (latest first)
    blobs.sort(key=lambda b: b.name, reverse=True)

    previous_blob_name = blobs[0].name
    previous_blob = container.get_blob_client(previous_blob_name)
    latest_blob = container.get_blob_client("latest.json")

    content = previous_blob.download_blob().readall()
    latest_blob.upload_blob(content, overwrite=True)

    return previous_blob_name



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

def find_talent_for_use_case_newfile(df_ureq, df_selfassessment, df_talent, df_eval, df_hist, df_assign):

    sentence_model = get_model()

    # --- Step 1: Build agg sentences ---
    if 'agg_sentences' not in df_ureq.columns:
        df_ureq['agg_sentences'] = (
            df_ureq['Responsibilities'] + " " +
            df_ureq['Skill 1'].fillna('') + " " +
            df_ureq['Skill 2'].fillna('')
        )

    # Step 2
    skillsets = [col for col in df_selfassessment.columns if col != 'UNIQUE ID']

    results = []
    for _, row in df_ureq.iterrows():
        corpus = [row['agg_sentences']] + skillsets
        embeddings = sentence_model.encode(corpus)
        sim_scores = cosine_similarity([embeddings[0]], embeddings[1:])[0]

        for i, skill in enumerate(skillsets):
            results.append([
                row['Responsibilities'],
                row['Skill 1'],
                row['Skill 2'],
                row['Role'],        # KEEP ONLY for metadata, not for talent role
                skill,
                sim_scores[i]
            ])

    df_results = pd.DataFrame(
        results,
        columns=['Responsibilities', 'Skill 1', 'Skill 2', 'Requested_Role', 'Skillset', 'Similarity score']
    )
    

    # FILTER
    df_results_filtered = df_results[df_results['Similarity score'] >= 0.3]

    # --- Step 4: Role Person removed; rely on ROLE from Talent ---
    unique_ids = df_selfassessment[['UNIQUE ID']].drop_duplicates()
    merged_df = df_results_filtered.merge(unique_ids, how='cross')

    # --- Step 5: Skill Score ---
    def get_skill_score(row):
        try:
            val = df_selfassessment.loc[df_selfassessment['UNIQUE ID'] == row['UNIQUE ID'], row['Skillset']]
            return float(val.values[0]) if not val.empty else np.nan
        except:
            return np.nan

    merged_df['Skill Score'] = merged_df.apply(get_skill_score, axis=1)

    # --- Step 6: Aggregation (REMOVE Role entirely here) ---
    df_search = merged_df.groupby(
        ['Responsibilities', 'UNIQUE ID', 'Requested_Role'], as_index=False
    ).agg(Avg_SkillScore=('Skill Score', 'mean'))


    # --- Step 7: Merge with Talent & Eval ---
    df_talent['Durasi Bulan'] = df_talent['LAMA KERJA BERJALAN'].apply(convert_to_months)
    df_agg_talent = pd.merge(df_talent, df_eval, on='UNIQUE ID')

    # scoring eval already Capability Score
    df_agg_talent['scoring_eval'] = df_agg_talent['Capability Score']

    # MERGE
    df_merged = pd.merge(df_search, df_agg_talent, on='UNIQUE ID')

    # --- Step 8: History ---
    df_hist_count = df_hist.groupby("UNIQUE ID")["PRODUCT / USECASE"].nunique().reset_index(name="job_count")
    df_final = df_merged.merge(df_hist_count, on='UNIQUE ID', how='left').fillna({'job_count': 0})

    # --- Step 9: Final scoring ---
    a, b, r = 0.47, 0.53, 1.2
    df_final['d'] = df_final['Avg_SkillScore'] * a + df_final['scoring_eval'] * b
    df_final['finalscore'] = df_final['d']
    df_final['finalscore_scaled'] = df_final.groupby('Responsibilities')['finalscore'].transform(minmax_scaling)

    # --- REMOVE "Role" BUT PRESERVE "Required_Role" FOR TRACEABILITY ---
    if "Role" in df_final.columns:
        df_final.drop(columns=["Role"], inplace=True)

    # Rename Required_Role → Requested_Role to avoid confusion
    df_final = df_final.rename(columns={"Required_Role": "Requested_Role"})

    return df_final


        
@app.route(route="talent_recommender_newfile", auth_level=func.AuthLevel.FUNCTION)
def talent_recommender_newfile(req: func.HttpRequest) -> func.HttpResponse:
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
        datasets = load_datasets_from_json_newfile(datasets_json)
        if not datasets:
            return func.HttpResponse(
                json.dumps({"error": "Could not load datasets. Check logs."}),
                status_code=500
            )
            
        for name, df in datasets.items():
            print(name, len(df), df.columns.tolist()[:10])

        # --- Run full pipeline (same as run_background_analysis) ---
        df_final = find_talent_for_use_case_newfile(
            df_ureq=datasets["df_ureq"],
            df_selfassessment=datasets["df_selfassessment"],
            df_talent=datasets["df_talent"],
            df_eval=datasets["df_eval"],
            df_hist=datasets["df_hist"],
            df_assign=datasets["df_assign"]
        )

        # Optional: reorder or limit columns (to mirror your previous final_df)
        columns_order = [
            "Responsibilities", "Skill 1", "Skill 2", "Requested_Role",
            "agg_sentences", "UNIQUE ID", "Avg_SkillScore", 
            "Cluster", "ROLE", "LAMA KERJA BERJALAN", "GRADE", "LEVEL", "LG INDEX",
            "Durasi Bulan", "Expert Judgement", "Capability Score", 
            "scoring_eval", "job_count", "d", "finalscore", "finalscore_scaled"
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

        # Convert to DataFrame for validation & cleaning
        df = pd.DataFrame(data_list)

        # Deduplicate
        if not df.empty:
            df = df.drop_duplicates(subset=["UNIQUE ID", "Responsibilities"], keep="first")

        # --- VALIDATE BEFORE UPLOADING ---
        if not is_json_valid(df):
            # INVALID → perform automatic rollback
            blob_service = BlobServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])
            container = blob_service.get_container_client("blobcleancontainer")

            restored_from = restore_previous_version(container)

            return func.HttpResponse(
                json.dumps({
                    "error": "Invalid data produced. Rolled back to previous version.",
                    "restored_from": restored_from
                }),
                mimetype="application/json",
                status_code=400
            )

        # --- DATA IS VALID → proceed with normal upload ---
        clean_json = df.to_json(orient="records")

        blob_conn_str = os.environ["AzureWebJobsStorage"]
        blob_service = BlobServiceClient.from_connection_string(blob_conn_str)
        container_name = "blobcleancontainer"
        container = blob_service.get_container_client(container_name)

        try:
            container.create_container()
        except Exception as e:
            if "ContainerAlreadyExists" not in str(e):
                raise

        # Save timestamped version
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        blob_name = f"talent_results_{timestamp}.json"
        blob_client = container.get_blob_client(blob_name)
        blob_client.upload_blob(clean_json, overwrite=True)

        # Save/overwrite latest.json
        latest_blob = container.get_blob_client("latest.json")
        latest_blob.upload_blob(clean_json, overwrite=True)

        return func.HttpResponse(
            json.dumps({
                "message": "Upload successful.",
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
