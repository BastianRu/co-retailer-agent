import pandas as pd
import boto3
import io
from typing import Dict
import os
from dotenv import load_dotenv
import time

load_dotenv()

#Config (remember move to .env later)
BUCKET_NAME = "co-retailer-agent-2026"   # bucket-name
PREFIX = ""                              # "" if root

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "us-east-2")
)

# In-memory cache: table_name -> DataFrame
_CACHE: Dict[str, pd.DataFrame] = {}


#loads ALL the .csv files from the bucket (used for warm-up method)
#skips tables already in cache
def load_all_s3_data() -> Dict[str, pd.DataFrame]:
    global _CACHE

    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)

    for obj in response.get('Contents', []):
        key = obj['Key']
        if not key.endswith('.csv'):
            continue

        table_name = key.replace(PREFIX, '').replace('.csv', '').replace('/', '')

        if table_name in _CACHE:
            continue

        obj_response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        _CACHE[table_name] = pd.read_csv(io.BytesIO(obj_response['Body'].read()))

    print(f"\n Loaded tables: {len(_CACHE)}")
    return dict(_CACHE)


#returns a single table, fetching from S3 only on first access
def load_s3_data(key: str) -> pd.DataFrame:
    global _CACHE

    if key not in _CACHE:
        obj_response = s3_client.get_object(
            Bucket=BUCKET_NAME,
            Key=f"{PREFIX}{key}.csv"
        )
        _CACHE[key] = pd.read_csv(io.BytesIO(obj_response['Body'].read()))

    return _CACHE[key]

#tests
if __name__ == "__main__":
    _start = time.perf_counter()
    data = load_s3_data("customers")
    #data2 = load_all_s3_data()

    resultado = data.head()
    elapsed = time.perf_counter() - _start
    print(resultado[resultado["phone"] == "+57 300 133 8908"  ] )
    print(f"{elapsed:.3f}s\n")
    #Provisional results:
    #      - ~2.6s cold start (with warmp_up)
    #      - ~0.6-1.1s (1.1s for the longest table (tracking)) 
