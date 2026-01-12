from openai import OpenAI
import re
import time
import psutil
from pathlib import Path
from datetime import datetime
import json
import shutil
import os
import tempfile
from botocore.exceptions import ClientError
import boto3
from pyspark.sql import SparkSession
import streamlit as st
from dotenv import load_dotenv
load_dotenv()


def update_env_file(key, value):
    """
    Updates a key-value pair in the .env file.
    Creates the .env file if it doesn't exist.
    """
    env_file = '.env'
    lines = []
    key_found = False

    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            lines = f.readlines()

    with open(env_file, 'w') as f:
        for line in lines:
            if line.strip().startswith(f'{key}='):
                f.write(f'{key}="{value}"\n')
                key_found = True
            else:
                f.write(line)
        if not key_found:
            f.write(f'{key}="{value}"\n')


# --- CONFIGURATION & PATHS ---
CONFIG_FILE = "config.json"
STORAGE_FILE = "spark_tables_metadata.json"
AWS_CREDENTIALS_ENV_PREFIX = "AWS_"
AZURE_CREDENTIALS_ENV_PREFIX = "AZURE_"
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
PROMPTS_FILE = "prompts.json"
DOWNLOADS_DIR = "spark_downloads"
os.makedirs(DOWNLOADS_DIR, exist_ok=True)
TEMP_DATA_DIR = "temp_data"
os.makedirs(TEMP_DATA_DIR, exist_ok=True)

# Configuration files
STREAMLIT_CONFIG_DIR = ".streamlit"
os.makedirs(STREAMLIT_CONFIG_DIR, exist_ok=True)
STREAMLIT_CONFIG_FILE = Path(".streamlit/config.toml")


# --- CONFIG LOADERS ---
def load_app_config():
    """Load UI settings like Title and Headers"""
    default = {
        "app_title": "S3 Spark Tool",
        "header_title": "🔍 S3 Spark SQL Query Tool",
        "sidebar_header": "⚙️ Settings",
        "ai_section_title": "🤖 AI Data Analysis"
    }
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading or parsing {CONFIG_FILE}: {e}")
        pass
    return default


# --- INITIALIZATION ---
ui_cfg = load_app_config()
st.set_page_config(page_title=ui_cfg.get("app_title"), layout="wide")


def save_openai_key(api_key):
    """(Removed direct file saving) Advise user to set environment variable instead."""
    os.environ[OPENAI_API_KEY_ENV_VAR] = api_key
    return True


def load_openai_key():
    """Load OpenAI API key from environment variable"""
    return os.getenv(OPENAI_API_KEY_ENV_VAR)


def validate_openai_key(api_key):
    """Validate OpenAI API key"""
    try:
        client = OpenAI(api_key=api_key)
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        return True, "API key is valid!"
    except Exception as e:
        return False, f"Invalid API key: {str(e)}"


def get_table_summary_spark(table_name, credentials):
    """Get comprehensive summary of table for AI context using pure Spark"""
    try:
        table_info = st.session_state['tables'][table_name]
        spark = get_spark_session(credentials)
        df = table_info['dataframe']

        # Get sample data using Spark
        sample_rows = df.limit(5).collect()
        sample_data = "\n".join([str(row.asDict()) for row in sample_rows])

        # Get numeric columns
        numeric_cols = [f.name for f in df.schema.fields if str(f.dataType) in [
            'IntegerType', 'LongType', 'FloatType', 'DoubleType', 'DecimalType']]

        stats_summary = ""
        if numeric_cols:
            stats_df = df.select(numeric_cols).summary(
                "count", "mean", "stddev", "min", "max")
            stats_rows = stats_df.collect()
            stats_summary = "\n".join([str(row.asDict())
                                      for row in stats_rows])

        summary = f"""
TABLE: {table_name}
Total Rows: {table_info['row_count']:,}
Total Columns: {table_info['col_count']}

SCHEMA:
{chr(10).join([f"- {col['name']}: {col['type']}" for col in table_info['schema']])}

SAMPLE DATA (first 5 rows):
{sample_data}

NUMERIC COLUMN STATISTICS:
{stats_summary if stats_summary else "No numeric columns"}
"""
        return summary
    except Exception as e:
        return f"Error getting table summary: {str(e)}"


def execute_query_for_ai(query, credentials):
    """Execute a SQL query and return results as string using Spark"""
    try:
        spark = get_spark_session(credentials)
        result = spark.sql(query)
        rows = result.limit(100).collect()
        result_str = "\n".join([str(row.asDict()) for row in rows])
        return True, result_str, len(rows)
    except Exception as e:
        return False, str(e), 0


# --- PROMPT ENGINE ---
def load_prompts():
    """Load AI prompts from external JSON file with fallback defaults"""
    default_prompts = {
        "system_message": "You are a professional Data Analyst with expertise in SQL and data analysis.",
        "one_shot_example": "Example: When asked 'What are the top 5 products by revenue?', analyze the schema and provide: 'Based on the data, I'll write a query to find the top 5 products:\n```sql\nSELECT product_name, SUM(revenue) as total_revenue\nFROM sales_table\nGROUP BY product_name\nORDER BY total_revenue DESC\nLIMIT 5\n```\nThis query groups sales by product and returns the highest earners.'",
        "analysis_instruction": "Analyze this table: {table_name}\n\nUser Question: {user_question}\n\nProvide a clear answer. If you need to query the data, include a SQL query in a ```sql code block.",
        "results_interpretation": "Query Results:\n{query_result}\n\nOriginal Question: {user_question}\n\nInterpret these results and provide actionable insights."
    }
    try:
        if os.path.exists(PROMPTS_FILE):
            with open(PROMPTS_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading prompts.json: {e}")
    return default_prompts


# --- AI LOGIC (REFACTORED & FIXED) ---
def analyze_data_with_ai(table_name, user_question, credentials, api_key, stream=False):
    """
    Analyzes data using OpenAI's GPT-4o-mini model.
    Uses pure Spark instead of pandas
    """
    try:
        client = OpenAI(api_key=api_key)
        prompts = load_prompts()

        # Get richer table context
        df = st.session_state['tables'][table_name]['dataframe']
        table_info = st.session_state['tables'][table_name]

        # Build comprehensive context using Spark
        schema_str = "\n".join(
            [f"  - {col['name']}: {col['type']}" for col in table_info['schema']])

        # Get sample data using Spark collect
        sample_rows = df.limit(10).collect()
        sample_str = "\n".join([str(row.asDict()) for row in sample_rows])

        context = f"""Table: {table_name}
Total Rows: {table_info['row_count']:,}
Total Columns: {table_info['col_count']}

Schema:
{schema_str}

Sample Data (first 10 rows):
{sample_str}
"""

        # Build messages
        system_msg = f"{prompts['system_message']}\n\nAVAILABLE DATA:\n{context}"
        if prompts.get("one_shot_example"):
            system_msg += f"\n\n{prompts['one_shot_example']}"

        user_msg = prompts["analysis_instruction"].format(
            table_name=table_name,
            user_question=user_question
        )

        if stream:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.7,
                stream=True
            )
        else:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
    except Exception as e:
        if stream:
            yield {"error": str(e)}
        else:
            return f"Error: {str(e)}"


def analyze_with_query_results(table_name, user_question, query_result, credentials, api_key):
    """
    Performs follow-up analysis on SQL query results using OpenAI's GPT-4o-mini model.
    """
    try:
        client = OpenAI(api_key=api_key)
        prompts = load_prompts()

        system_prompt = prompts["results_interpretation"].format(
            query_result=query_result,
            user_question=user_question
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def parse_aws_credentials(credential_text):
    """Parse AWS credentials from export format"""
    try:
        credentials = {}
        patterns = {
            'access_key': r'export\s+AWS_ACCESS_KEY_ID\s*=\s*["\']([^"\']+)["\']',
            'secret_key': r'export\s+AWS_SECRET_ACCESS_KEY\s*=\s*["\']([^"\']+)["\']',
            'session_token': r'export\s+AWS_SESSION_TOKEN\s*=\s*["\']([^"\']+)["\']',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, credential_text,
                              re.IGNORECASE | re.MULTILINE)
            if match:
                credentials[key] = match.group(1)

        if 'access_key' in credentials and 'secret_key' in credentials:
            return credentials
        else:
            return None
    except Exception as e:
        return None


def parse_azure_credentials(credential_text):
    """Parse Azure credentials from export format"""
    try:
        credentials = {}
        patterns = {
            'account_name': r'export\s+AZURE_STORAGE_ACCOUNT\s*=\s*["\']([^"\']+)["\']',
            'account_key': r'export\s+AZURE_STORAGE_KEY\s*=\s*["\']([^"\']+)["\']',
            'container': r'export\s+AZURE_CONTAINER\s*=\s*["\']([^"\']+)["\']',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, credential_text,
                              re.IGNORECASE | re.MULTILINE)
            if match:
                credentials[key] = match.group(1)

        if 'account_name' in credentials and 'account_key' in credentials:
            return credentials
        else:
            return None
    except Exception as e:
        return None


def save_credentials(credentials, cloud_type='aws'):
    """Save cloud credentials to environment variables for the current session."""
    try:
        if cloud_type == 'aws':
            os.environ[f"{AWS_CREDENTIALS_ENV_PREFIX}ACCESS_KEY_ID"] = credentials.get(
                'access_key', '')
            os.environ[f"{AWS_CREDENTIALS_ENV_PREFIX}SECRET_ACCESS_KEY"] = credentials.get(
                'secret_key', '')
            if 'session_token' in credentials:
                os.environ[f"{AWS_CREDENTIALS_ENV_PREFIX}SESSION_TOKEN"] = credentials.get(
                    'session_token', '')
            else:
                if f"{AWS_CREDENTIALS_ENV_PREFIX}SESSION_TOKEN" in os.environ:
                    del os.environ[f"{AWS_CREDENTIALS_ENV_PREFIX}SESSION_TOKEN"]
        elif cloud_type == 'azure':
            os.environ[f"{AZURE_CREDENTIALS_ENV_PREFIX}STORAGE_ACCOUNT"] = credentials.get(
                'account_name', '')
            os.environ[f"{AZURE_CREDENTIALS_ENV_PREFIX}STORAGE_KEY"] = credentials.get(
                'account_key', '')
            if 'container' in credentials:
                os.environ[f"{AZURE_CREDENTIALS_ENV_PREFIX}CONTAINER"] = credentials.get(
                    'container', '')
        return True
    except Exception as e:
        st.error(f"Error saving credentials: {e}")
        return False


def load_credentials(cloud_type='aws'):
    """Load cloud credentials from environment variables."""
    if cloud_type == 'aws':
        access_key = os.getenv(f"{AWS_CREDENTIALS_ENV_PREFIX}ACCESS_KEY_ID")
        secret_key = os.getenv(
            f"{AWS_CREDENTIALS_ENV_PREFIX}SECRET_ACCESS_KEY")
        session_token = os.getenv(
            f"{AWS_CREDENTIALS_ENV_PREFIX}SESSION_TOKEN")

        if access_key and secret_key:
            return {
                'access_key': access_key,
                'secret_key': secret_key,
                'session_token': session_token,
                'type': 'aws'
            }
    elif cloud_type == 'azure':
        account_name = os.getenv(
            f"{AZURE_CREDENTIALS_ENV_PREFIX}STORAGE_ACCOUNT")
        account_key = os.getenv(f"{AZURE_CREDENTIALS_ENV_PREFIX}STORAGE_KEY")
        container = os.getenv(f"{AZURE_CREDENTIALS_ENV_PREFIX}CONTAINER")

        if account_name and account_key:
            return {
                'account_name': account_name,
                'account_key': account_key,
                'container': container,
                'type': 'azure'
            }
    return None


def validate_s3_credentials(credentials):
    """Validate AWS credentials by attempting to list S3 buckets"""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=credentials.get('access_key'),
            aws_secret_access_key=credentials.get('secret_key'),
            aws_session_token=credentials.get('session_token'),
            region_name='us-east-1'
        )
        s3_client.list_buckets()
        return True, "Credentials are valid!"
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'InvalidAccessKeyId':
            return False, "Invalid Access Key ID"
        elif error_code == 'SignatureDoesNotMatch':
            return False, "Invalid Secret Access Key"
        elif error_code == 'ExpiredToken':
            return False, "Session token has expired"
        else:
            return False, f"Validation failed: {error_code}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_azure_credentials(credentials):
    """Validate Azure credentials by attempting to connect to storage"""
    try:
        from azure.storage.blob import BlobServiceClient
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={credentials.get('account_name')};AccountKey={credentials.get('account_key')};EndpointSuffix=core.windows.net"
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string)
        # Try to list containers
        list(blob_service_client.list_containers(max_results=1))
        return True, "Azure credentials are valid!"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def apply_theme(theme_mode):
    """Apply theme by modifying Streamlit config and page config"""
    try:
        # Read existing config if it exists
        existing_config = ""
        if STREAMLIT_CONFIG_FILE.exists():
            with open(STREAMLIT_CONFIG_FILE, 'r') as f:
                existing_config = f.read()

        # Update or add theme section
        theme_config = f"""
[theme]
base = "{theme_mode}"
primaryColor = "#FF4B4B"
"""

        # Remove old theme section if exists
        if '[theme]' in existing_config:
            lines = existing_config.split('\n')
            new_lines = []
            skip = False
            for line in lines:
                if line.strip().startswith('[theme]'):
                    skip = True
                    continue
                elif line.strip().startswith('[') and skip:
                    skip = False
                if not skip:
                    new_lines.append(line)
            existing_config = '\n'.join(new_lines)

        # Write config
        with open(STREAMLIT_CONFIG_FILE, 'w') as f:
            f.write(existing_config + '\n' + theme_config)

        return True
    except Exception as e:
        st.error(f"Error applying theme: {e}")
        return False


@st.cache_resource
def get_spark_session(credentials):
    """
    Creates and configures a SparkSession with cloud credentials.
    Uses st.cache_resource to ensure the SparkSession is created only once.
    """
    builder = SparkSession.builder.appName(ui_cfg.get("app_title")) \
        .config("spark.local.dir", os.path.join(os.getcwd(), TEMP_DATA_DIR, "spark"))

    if credentials:
        if credentials.get('type') == 'aws':
            builder = builder \
                .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
                .config("spark.hadoop.fs.s3a.access.key", credentials.get('access_key', '')) \
                .config("spark.hadoop.fs.s3a.secret.key", credentials.get('secret_key', '')) \
                .config("spark.hadoop.fs.s3a.session.token", credentials.get('session_token', '')) \
                .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        elif credentials.get('type') == 'azure':
            builder = builder \
                .config("spark.jars.packages", "org.apache.hadoop:hadoop-azure:3.3.4,com.microsoft.azure:azure-storage:8.6.6") \
                .config(f"spark.hadoop.fs.azure.account.key.{credentials.get('account_name')}.blob.core.windows.net", credentials.get('account_key', ''))

    return builder \
        .config("spark.ui.enabled", "true") \
        .config("spark.ui.port", "4040") \
        .getOrCreate()


def get_system_resources():
    """Get system memory and disk usage - Application usage vs Available"""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    # Get Spark process memory
    process = psutil.Process(os.getpid())
    app_mem = process.memory_info().rss

    # Calculate app disk usage (downloads + temp data)
    app_disk_usage = 0
    for folder in [DOWNLOADS_DIR, TEMP_DATA_DIR]:
        if os.path.exists(folder):
            for dirpath, dirnames, filenames in os.walk(folder):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        app_disk_usage += os.path.getsize(filepath)
                    except:
                        pass

    return {
        'app_mem_gb': app_mem / (1024**3),
        'mem_available_gb': mem.available / (1024**3),
        'mem_total_gb': mem.total / (1024**3),
        'mem_percent': (app_mem / mem.available) * 100 if mem.available > 0 else 0,
        'app_disk_gb': app_disk_usage / (1024**3),
        'disk_available_gb': disk.free / (1024**3),
        'disk_total_gb': disk.total / (1024**3),
        'disk_percent': (app_disk_usage / disk.free) * 100 if disk.free > 0 else 0,
    }


def format_bytes(b):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if b < 1024:
            return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} PB"


def save_metadata():
    try:
        metadata = {}
        for name, info in st.session_state.get('tables', {}).items():
            metadata[name] = {k: v for k, v in info.items()
                              if k not in ['dataframe', 'temp_dir']}
        with open(STORAGE_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving: {e}")
        return False


def load_metadata():
    try:
        if os.path.exists(STORAGE_FILE):
            with open(STORAGE_FILE) as f:
                return json.load(f)
    except:
        pass
    return {}


def delete_table_metadata(table_name):
    """Delete table from metadata and clean up downloaded files"""
    try:
        metadata = load_metadata()
        if table_name in metadata:
            table_info = metadata[table_name]
            local_files = table_info.get('local_files', [])
            for file_path in local_files:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass

            local_path = table_info.get('local_path', '')
            if local_path and os.path.exists(local_path):
                try:
                    if not os.listdir(local_path):
                        os.rmdir(local_path)
                except:
                    pass

            del metadata[table_name]
            with open(STORAGE_FILE, 'w') as f:
                json.dump(metadata, f, indent=2)
            return True, f"Deleted {table_name} and cleaned up {len(local_files)} file(s)"
        return False, "Table not found in metadata"
    except Exception as e:
        return False, str(e)


def download_from_s3_with_progress(s3_path, file_format, table_name, credentials):
    """Download files from S3 with real-time progress tracking"""
    try:
        if s3_path.startswith('s3://'):
            s3_path = s3_path[5:]
        elif s3_path.startswith('s3a://'):
            s3_path = s3_path[6:]

        s3_path = s3_path.rstrip('/')
        parts = s3_path.split('/')
        bucket_name = parts[0]
        key = '/'.join(parts[1:]) if len(parts) > 1 else ''

        s3_client = boto3.client(
            's3',
            aws_access_key_id=credentials.get('access_key'),
            aws_secret_access_key=credentials.get('secret_key'),
            aws_session_token=credentials.get('session_token'),
            region_name='us-east-1'
        )

        valid_extensions = {
            'parquet': ['.parquet', '.parq'],
            'csv': ['.csv', '.tsv', '.txt']
        }

        table_download_dir = os.path.join(DOWNLOADS_DIR, table_name)
        os.makedirs(table_download_dir, exist_ok=True)

        all_files = []
        total_size = 0

        try:
            response = s3_client.head_object(Bucket=bucket_name, Key=key)
            file_ext = os.path.splitext(key)[1].lower()
            if file_ext in valid_extensions[file_format]:
                all_files.append(
                    {'key': key, 'size': response['ContentLength']})
                total_size = response['ContentLength']
        except ClientError:
            prefix = key + '/' if key else ''
            paginator = s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        obj_key = obj['Key']
                        if obj_key.endswith('/') or obj_key.split('/')[-1].startswith('.'):
                            continue
                        file_ext = os.path.splitext(obj_key)[1].lower()
                        if file_ext in valid_extensions[file_format]:
                            all_files.append(
                                {'key': obj_key, 'size': obj['Size']})
                            total_size += obj['Size']

        if not all_files:
            return False, f"No {file_format} files found", [], 0, table_download_dir

        downloaded_files = []
        downloaded_size = 0

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, file_info in enumerate(all_files, 1):
            obj_key = file_info['key']
            file_size = file_info['size']
            filename = os.path.basename(obj_key)
            local_path = os.path.join(table_download_dir, filename)

            s3_client.download_file(bucket_name, obj_key, local_path)
            downloaded_files.append(local_path)
            downloaded_size += file_size

            progress = downloaded_size / total_size
            progress_bar.progress(progress)
            status_text.text(
                f"📥 Downloaded: {format_bytes(downloaded_size)} / {format_bytes(total_size)} ({idx}/{len(all_files)} files)")

        progress_bar.empty()
        status_text.empty()

        return True, f"Downloaded {len(downloaded_files)} file(s)", downloaded_files, total_size, table_download_dir

    except Exception as e:
        return False, f"Error: {str(e)}", [], 0, ""


def load_from_s3_direct(s3_path, file_format, csv_options, credentials):
    try:
        spark = get_spark_session(credentials)
        if s3_path.startswith('s3://'):
            s3_path = 's3a://' + s3_path[5:]

        if file_format == "parquet":
            df = spark.read.parquet(s3_path)
        else:
            reader = spark.read.format("csv")
            if csv_options:
                for k, v in csv_options.items():
                    reader = reader.option(k, v)
            df = reader.load(s3_path)
        return True, df, None
    except Exception as e:
        return False, None, str(e)


def load_from_local_files(local_files, file_format, csv_options, credentials):
    """Load data from local files"""
    try:
        spark = get_spark_session(credentials)
        if file_format == "parquet":
            df = spark.read.parquet(*local_files)
        else:
            reader = spark.read.format("csv")
            if csv_options:
                for k, v in csv_options.items():
                    reader = reader.option(k, v)
            df = reader.load(*local_files)
        return True, df, None
    except Exception as e:
        return False, None, str(e)


def restore_table(table_name, meta, credentials):
    """Restore table from metadata"""
    try:
        spark = get_spark_session(credentials)
        local_files = meta.get('local_files', [])
        files_exist = all(os.path.exists(f)
                          for f in local_files) if local_files else False

        if files_exist and local_files:
            success, df, error = load_from_local_files(
                local_files, meta['file_format'], meta.get('csv_options', {}), credentials)
            if not success:
                return False, f"Could not load from local files: {error}"
            source_type = "Local Cache"
        elif meta['source'].startswith('S3:'):
            success, df, error = load_from_s3_direct(
                meta['s3_path'], meta['file_format'], meta.get('csv_options', {}), credentials)
            if not success:
                return False, f"Could not restore from S3: {error}"
            source_type = "S3 Direct"
        else:
            return False, "Cannot restore: No local files and not an S3 table"

        df.cache()
        df.createOrReplaceTempView(table_name)

        st.session_state['tables'][table_name] = {
            'dataframe': df,
            'temp_dir': None,
            'loaded': True,
            **{k: v for k, v in meta.items()}
        }

        return True, f"Restored from {source_type}"
    except Exception as e:
        return False, str(e)


def show_openai_setup():
    """Show OpenAI setup dialog, allowing user to enter key and save to .env file."""
    st.title(ui_cfg.get("header_title"))
    st.markdown("### Enable AI-Powered Data Analysis")

    # Add back button
    if st.button("← Back to Main", key="back_from_openai"):
        st.session_state['show_openai_setup'] = False
        st.rerun()

    api_key_from_env = load_openai_key()
    if api_key_from_env:
        st.success(
            f"✅ OpenAI API Key loaded from environment variable '{OPENAI_API_KEY_ENV_VAR}'.")
    else:
        st.warning(
            f"Waiting for '{OPENAI_API_KEY_ENV_VAR}' environment variable to be set.")

    with st.expander("Enter/Update OpenAI API Key", expanded=not api_key_from_env):
        st.warning("⚠️ **Security Warning:** Saving credentials from the UI will write to a local `.env` file. It is more secure to set environment variables directly in your shell or operating system.")

        api_key = st.text_input(
            "OpenAI API Key:", type="password", placeholder="sk-...")

        if st.button("💾 Save to .env and Validate"):
            if api_key.strip():
                with st.spinner("Validating and saving API key..."):
                    update_env_file(OPENAI_API_KEY_ENV_VAR, api_key)
                    load_dotenv(override=True)
                    is_valid, message = validate_openai_key(api_key)
                    if is_valid:
                        st.session_state['openai_key'] = api_key
                        st.session_state['show_openai_setup'] = False
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
            else:
                st.error("Please enter an API key")

    if st.button("⏭️ Skip (Disable AI Features)", key="skip_ai_setup"):
        st.session_state['openai_key'] = None
        st.session_state['show_openai_setup'] = False
        st.rerun()


def show_credentials_setup():
    """Show Cloud credentials setup interface, allowing user to paste or enter manually."""
    st.title("🔐 Cloud Credentials Setup")

    # Add back button
    if st.button("← Back to Main", key="back_from_creds"):
        st.session_state['skip_credentials'] = True
        st.rerun()

    # Cloud provider selection
    cloud_provider = st.radio(
        "Select Cloud Provider:", ["AWS", "Azure"], horizontal=True, key="cloud_provider_select")

    if cloud_provider == "AWS":
        credentials = load_credentials('aws')
        if credentials:
            st.success(f"✅ AWS Credentials loaded from environment variables.")
        else:
            st.warning(
                f"Waiting for AWS credentials environment variables to be set (e.g., {AWS_CREDENTIALS_ENV_PREFIX}ACCESS_KEY_ID).")

        with st.expander("Enter/Update AWS Credentials", expanded=not credentials):
            st.warning("⚠️ **Security Warning:** Saving credentials from the UI will write to a local `.env` file. It is more secure to set environment variables directly in your shell or operating system.")

            input_method = st.radio("Choose input method:", [
                                    "Paste Credentials", "Enter Manually"], key="aws_input_method")

            if input_method == "Paste Credentials":
                credentials_text = st.text_area(
                    "Paste your AWS credentials here:",
                    height=200,
                    placeholder="""export AWS_ACCESS_KEY_ID="ASIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."  # Optional for temporary credentials""",
                    key="aws_paste"
                )

                if st.button("💾 Save to .env and Validate", key="aws_paste_save"):
                    if credentials_text.strip():
                        parsed_creds = parse_aws_credentials(credentials_text)
                        if parsed_creds:
                            with st.spinner("Validating and saving credentials..."):
                                update_env_file(
                                    f"{AWS_CREDENTIALS_ENV_PREFIX}ACCESS_KEY_ID", parsed_creds['access_key'])
                                update_env_file(
                                    f"{AWS_CREDENTIALS_ENV_PREFIX}SECRET_ACCESS_KEY", parsed_creds['secret_key'])
                                if 'session_token' in parsed_creds:
                                    update_env_file(
                                        f"{AWS_CREDENTIALS_ENV_PREFIX}SESSION_TOKEN", parsed_creds['session_token'])

                                load_dotenv(override=True)

                                is_valid, message = validate_s3_credentials(
                                    parsed_creds)

                                if is_valid:
                                    parsed_creds['type'] = 'aws'
                                    st.session_state['credentials'] = parsed_creds
                                    st.session_state['credentials_validated'] = True
                                    st.session_state['cloud_type'] = 'aws'
                                    st.success(f"✅ {message}")
                                    st.rerun()
                                else:
                                    st.error(f"❌ {message}")
                        else:
                            st.error(
                                "Could not parse credentials. Please check the format.")
                    else:
                        st.error("Please paste your credentials.")

            else:
                aws_access_key = st.text_input(
                    "AWS Access Key ID:", placeholder="ASIA...", key="aws_access_key")
                aws_secret_key = st.text_input(
                    "AWS Secret Access Key:", type="password", key="aws_secret_key")
                aws_session_token = st.text_input(
                    "AWS Session Token (optional):", type="password", key="aws_session_token")

                if st.button("💾 Save to .env and Validate", key="aws_manual_save"):
                    if aws_access_key.strip() and aws_secret_key.strip():
                        new_creds = {
                            'access_key': aws_access_key,
                            'secret_key': aws_secret_key,
                            'session_token': aws_session_token if aws_session_token.strip() else None,
                            'type': 'aws'
                        }

                        with st.spinner("Validating and saving credentials..."):
                            update_env_file(
                                f"{AWS_CREDENTIALS_ENV_PREFIX}ACCESS_KEY_ID", aws_access_key)
                            update_env_file(
                                f"{AWS_CREDENTIALS_ENV_PREFIX}SECRET_ACCESS_KEY", aws_secret_key)
                            if aws_session_token.strip():
                                update_env_file(
                                    f"{AWS_CREDENTIALS_ENV_PREFIX}SESSION_TOKEN", aws_session_token)

                            load_dotenv(override=True)

                            is_valid, message = validate_s3_credentials(
                                new_creds)

                            if is_valid:
                                st.session_state['credentials'] = new_creds
                                st.session_state['credentials_validated'] = True
                                st.session_state['cloud_type'] = 'aws'
                                st.success(f"✅ {message}")
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                    else:
                        st.error(
                            "Please enter both AWS Access Key ID and Secret Access Key.")

    else:  # Azure
        credentials = load_credentials('azure')
        if credentials:
            st.success(
                f"✅ Azure Credentials loaded from environment variables.")
        else:
            st.warning(
                f"Waiting for Azure credentials environment variables to be set (e.g., {AZURE_CREDENTIALS_ENV_PREFIX}STORAGE_ACCOUNT).")

        with st.expander("Enter/Update Azure Credentials", expanded=not credentials):
            st.warning("⚠️ **Security Warning:** Saving credentials from the UI will write to a local `.env` file. It is more secure to set environment variables directly in your shell or operating system.")

            input_method = st.radio("Choose input method:", [
                                    "Paste Credentials", "Enter Manually"], key="azure_input_method")

            if input_method == "Paste Credentials":
                credentials_text = st.text_area(
                    "Paste your Azure credentials here:",
                    height=200,
                    placeholder="""export AZURE_STORAGE_ACCOUNT="mystorageaccount"
export AZURE_STORAGE_KEY="..."
export AZURE_CONTAINER="mycontainer"  # Optional""",
                    key="azure_paste"
                )

                if st.button("💾 Save to .env and Validate", key="azure_paste_save"):
                    if credentials_text.strip():
                        parsed_creds = parse_azure_credentials(
                            credentials_text)
                        if parsed_creds:
                            with st.spinner("Validating and saving credentials..."):
                                update_env_file(
                                    f"{AZURE_CREDENTIALS_ENV_PREFIX}STORAGE_ACCOUNT", parsed_creds['account_name'])
                                update_env_file(
                                    f"{AZURE_CREDENTIALS_ENV_PREFIX}STORAGE_KEY", parsed_creds['account_key'])
                                if 'container' in parsed_creds:
                                    update_env_file(
                                        f"{AZURE_CREDENTIALS_ENV_PREFIX}CONTAINER", parsed_creds['container'])

                                load_dotenv(override=True)

                                is_valid, message = validate_azure_credentials(
                                    parsed_creds)

                                if is_valid:
                                    parsed_creds['type'] = 'azure'
                                    st.session_state['credentials'] = parsed_creds
                                    st.session_state['credentials_validated'] = True
                                    st.session_state['cloud_type'] = 'azure'
                                    st.success(f"✅ {message}")
                                    st.rerun()
                                else:
                                    st.error(f"❌ {message}")
                        else:
                            st.error(
                                "Could not parse credentials. Please check the format.")
                    else:
                        st.error("Please paste your credentials.")

            else:
                azure_account_name = st.text_input(
                    "Azure Storage Account Name:", placeholder="mystorageaccount", key="azure_account")
                azure_account_key = st.text_input(
                    "Azure Storage Key:", type="password", key="azure_key")
                azure_container = st.text_input(
                    "Azure Container (optional):", placeholder="mycontainer", key="azure_container")

                if st.button("💾 Save to .env and Validate", key="azure_manual_save"):
                    if azure_account_name.strip() and azure_account_key.strip():
                        new_creds = {
                            'account_name': azure_account_name,
                            'account_key': azure_account_key,
                            'container': azure_container if azure_container.strip() else None,
                            'type': 'azure'
                        }

                        with st.spinner("Validating and saving credentials..."):
                            update_env_file(
                                f"{AZURE_CREDENTIALS_ENV_PREFIX}STORAGE_ACCOUNT", azure_account_name)
                            update_env_file(
                                f"{AZURE_CREDENTIALS_ENV_PREFIX}STORAGE_KEY", azure_account_key)
                            if azure_container.strip():
                                update_env_file(
                                    f"{AZURE_CREDENTIALS_ENV_PREFIX}CONTAINER", azure_container)

                            load_dotenv(override=True)

                            is_valid, message = validate_azure_credentials(
                                new_creds)

                            if is_valid:
                                st.session_state['credentials'] = new_creds
                                st.session_state['credentials_validated'] = True
                                st.session_state['cloud_type'] = 'azure'
                                st.success(f"✅ {message}")
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                    else:
                        st.error(
                            "Please enter both Azure Storage Account Name and Key.")

    if st.button("⏭️ Skip (Local Files Only)", key="skip_cloud_setup"):
        st.session_state['credentials'] = None
        st.session_state['credentials_validated'] = False
        st.session_state['skip_credentials'] = True
        st.rerun()


def _display_conversation_entry(conv_num, conv_entry):
    """Helper function to display a single AI conversation entry."""
    with st.container():
        col1, col2 = st.columns([5, 1])
        with col1:
            if 'question' in conv_entry:
                st.markdown(f"**#{conv_num} - Table: {conv_entry['table']}**")
            elif 'query' in conv_entry:
                st.markdown(f"**#{conv_num} - Query Results Analysis**")
            elif 'type' in conv_entry and conv_entry['type'] == 'sql_analysis':
                st.markdown(f"**#{conv_num} - SQL Results Analysis**")

        with col2:
            expand_key = f"expand_conv_{conv_num}"
            if expand_key not in st.session_state:
                st.session_state[expand_key] = False

            if st.button("🔍" if not st.session_state[expand_key] else "🔼",
                         key=f"btn_expand_{conv_num}",
                         help="Expand/Collapse"):
                st.session_state[expand_key] = not st.session_state[expand_key]
                st.rerun()

        if 'question' in conv_entry:
            st.markdown(f"**Q:** {conv_entry['question']}")
            if st.session_state.get(expand_key, False):
                st.markdown("**A:**")
                st.markdown(conv_entry['response'])
            else:
                response_preview = conv_entry['response'][:300] + "..." if len(
                    conv_entry['response']) > 300 else conv_entry['response']
                st.markdown(f"**A:** {response_preview}")
        elif 'query' in conv_entry:
            st.code(conv_entry['query'], language="sql")
            if st.session_state.get(expand_key, False):
                st.markdown("**Analysis:**")
                st.markdown(conv_entry['analysis'])
            else:
                analysis_preview = conv_entry['analysis'][:300] + "..." if len(
                    conv_entry['analysis']) > 300 else conv_entry['analysis']
                st.markdown(f"**Analysis:** {analysis_preview}")
        elif 'type' in conv_entry and conv_entry['type'] == 'sql_analysis':
            st.code(conv_entry['query'], language="sql")
            if st.session_state.get(expand_key, False):
                st.markdown("**Full Analysis:**")
                st.markdown(conv_entry['analysis'])
            else:
                analysis_preview = conv_entry['analysis'][:300] + "..." if len(
                    conv_entry['analysis']) > 300 else conv_entry['analysis']
                st.markdown(f"**Analysis:** {analysis_preview}")

        st.markdown("---")


def dataframe_to_csv_string(df):
    """Convert Spark DataFrame to CSV string without using pandas"""
    rows = df.collect()
    columns = df.columns

    # Create CSV header
    csv_lines = [",".join(columns)]

    # Add data rows
    for row in rows:
        csv_lines.append(
            ",".join([str(val) if val is not None else "" for val in row]))

    return "\n".join(csv_lines)


def main():
    st.set_page_config(
        page_title="S3 Spark SQL Query Tool",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if 'tables' not in st.session_state:
        st.session_state['tables'] = {}
    if 'theme' not in st.session_state:
        st.session_state['theme'] = 'light'
    if 'metadata_cache' not in st.session_state:
        st.session_state['metadata_cache'] = load_metadata()
    if 'credentials' not in st.session_state:
        # Try to load from both AWS and Azure
        aws_creds = load_credentials('aws')
        azure_creds = load_credentials('azure')
        st.session_state['credentials'] = aws_creds or azure_creds
        st.session_state['cloud_type'] = 'aws' if aws_creds else (
            'azure' if azure_creds else None)
    if 'credentials_validated' not in st.session_state:
        st.session_state['credentials_validated'] = st.session_state['credentials'] is not None
    if 'openai_key' not in st.session_state:
        st.session_state['openai_key'] = load_openai_key()
    if 'ai_conversation' not in st.session_state:
        st.session_state['ai_conversation'] = []
    if 'show_openai_setup' not in st.session_state:
        st.session_state['show_openai_setup'] = False
    if 'staged_query' not in st.session_state:
        st.session_state['staged_query'] = None

    # Check if OpenAI setup is requested
    if st.session_state.get('show_openai_setup', False):
        show_openai_setup()
        return

    # Check if credentials are set up
    if st.session_state['credentials'] is None and 'skip_credentials' not in st.session_state:
        show_credentials_setup()
        return

    # Sidebar - Redesigned
    with st.sidebar:
        # Active Tables at Top
        if st.session_state['tables']:
            st.markdown("### 📊 Active Tables")
            for name, info in st.session_state['tables'].items():
                with st.expander(f"✅ {name}", expanded=False):
                    st.write(f"**Rows:** {info['row_count']:,}")
                    st.write(f"**Cols:** {info['col_count']}")
                    st.write(f"**Source:** {info['source']}")
                    if st.button(f"🗑️ Unload", key=f"unload_{name}", use_container_width=True):
                        del st.session_state['tables'][name]
                        try:
                            get_spark_session(
                                st.session_state['credentials']).catalog.dropTempView(name)
                        except:
                            pass
                        st.rerun()
            st.markdown("---")

        # Saved Tables Management
        st.markdown("### 📂 Saved Tables")
        metadata = st.session_state['metadata_cache']

        if metadata:
            st.info(f"Found {len(metadata)} saved table(s)")
            for table_name, meta in metadata.items():
                is_loaded = table_name in st.session_state['tables']
                has_local = bool(meta.get('local_files')) and all(
                    os.path.exists(f) for f in meta.get('local_files', []))

                if is_loaded:
                    status_icon = "✅"
                    status = "Loaded"
                elif has_local:
                    status_icon = "💾"
                    status = "Cached"
                else:
                    status_icon = "☁️"
                    status = "Cloud Only"

                with st.expander(f"{status_icon} {table_name} ({status})", expanded=False):
                    st.write(f"**Source:** {meta.get('source', 'Unknown')}")
                    st.write(f"**Rows:** {meta.get('row_count', 0):,}")
                    st.write(f"**Columns:** {meta.get('col_count', 0)}")

                    if has_local:
                        local_size = sum(os.path.getsize(f) for f in meta.get(
                            'local_files', []) if os.path.exists(f))
                        st.write(f"**Local Size:** {format_bytes(local_size)}")

                    col1, col2 = st.columns(2)
                    with col1:
                        if not is_loaded:
                            if st.button(f"📥 Load", key=f"load_{table_name}", use_container_width=True):
                                if st.session_state['credentials'] or has_local:
                                    with st.spinner(f"Loading {table_name}..."):
                                        success, msg = restore_table(
                                            table_name, meta, st.session_state['credentials'])
                                        if success:
                                            st.success(f"✅ {msg}")
                                            st.rerun()
                                        else:
                                            st.error(f"Error: {msg}")
                                else:
                                    st.error(
                                        "Cloud credentials required for cloud tables")
                        else:
                            st.success("✅ Loaded")

                    with col2:
                        if st.button(f"🗑️ Delete", key=f"delete_meta_{table_name}", use_container_width=True):
                            if is_loaded:
                                del st.session_state['tables'][table_name]
                                try:
                                    get_spark_session(
                                        st.session_state['credentials']).catalog.dropTempView(table_name)
                                except:
                                    pass
                            success, msg = delete_table_metadata(table_name)
                            if success:
                                st.session_state['metadata_cache'] = load_metadata(
                                )
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(f"Error: {msg}")
        else:
            st.caption("No saved tables found")

        st.markdown("---")

        # Action buttons
        if st.button("💾 Save Metadata", use_container_width=True):
            if save_metadata():
                st.session_state['metadata_cache'] = load_metadata()
                st.success("✅ Saved!")

        if st.button("🗑️ Clear All Active", use_container_width=True):
            st.session_state['tables'] = {}
            st.rerun()

        if st.button("🗑️ Clear Temporary Files", use_container_width=True):
            try:
                shutil.rmtree(TEMP_DATA_DIR)
                os.makedirs(TEMP_DATA_DIR, exist_ok=True)
                st.success("✅ Temporary files cleared!")
            except Exception as e:
                st.error(f"Error clearing temporary files: {e}")

        # Fixed icons at bottom with custom CSS for tight spacing
        st.markdown("---")
        st.markdown("""
            <style>
            div[data-testid="stHorizontalBlock"] > div {
                padding: 0px !important;
                margin: 0px !important;
            }
            div[data-testid="stHorizontalBlock"] button {
                padding: 0.25rem !important;
                margin: 0px !important;
            }
            </style>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            current_theme = st.session_state.get('theme', 'light')
            theme_icon = "🌙" if current_theme == 'light' else "☀️"
            if st.button(theme_icon, help="Toggle theme", use_container_width=True, key="theme_toggle"):
                new_theme = 'dark' if current_theme == 'light' else 'light'
                st.session_state['theme'] = new_theme
                if apply_theme(new_theme):
                    st.info(
                        "Theme preference saved! Please refresh the page (F5) to see the changes.")

        with col2:
            # Cloud status icon
            cloud_type = st.session_state.get('cloud_type', None)
            if st.session_state['credentials']:
                cloud_icon = "☁️✅" if cloud_type == 'aws' else "🔵✅"
                if cloud_type:
                    cloud_help = f"{cloud_type.upper()} Connected"
            else:
                cloud_icon = "☁️❌"
                cloud_help = "Not Connected"

            if st.button(cloud_icon, help=cloud_help, use_container_width=True, key="cloud_toggle"):
                if st.session_state['credentials']:
                    st.session_state['credentials'] = None
                    st.session_state['credentials_validated'] = False
                    st.session_state['cloud_type'] = None
                    if 'skip_credentials' in st.session_state:
                        del st.session_state['skip_credentials']
                    st.rerun()
                else:
                    if 'skip_credentials' in st.session_state:
                        del st.session_state['skip_credentials']
                    st.rerun()

        with col3:
            # AI status icon
            ai_icon = "🤖✅" if st.session_state['openai_key'] else "🤖❌"
            ai_help = "AI Enabled" if st.session_state['openai_key'] else "AI Disabled"

            if st.button(ai_icon, help=ai_help, use_container_width=True, key="ai_toggle"):
                if st.session_state['openai_key']:
                    st.session_state['openai_key'] = None
                st.session_state['show_openai_setup'] = True
                st.rerun()

        with col4:
            # System resources icon - App Used vs Available
            resources = get_system_resources()
            mem_used_pct = resources['mem_percent']

            # Use colors based on usage
            if mem_used_pct < 50:
                resource_icon = "💾🟢"
            elif mem_used_pct < 80:
                resource_icon = "💾🟡"
            else:
                resource_icon = "💾🔴"

            resource_help = f"App Memory: {resources['app_mem_gb']:.1f}GB / {resources['mem_available_gb']:.1f}GB available\nApp Disk: {resources['app_disk_gb']:.1f}GB / {resources['disk_available_gb']:.1f}GB available"

            st.button(
                resource_icon,
                help=resource_help,
                disabled=True,
                use_container_width=True,
                key="resources_display"
            )

    # Main area - rest of the code continues from original
    # (SQL Query section, AI Analysis section, Load Data section remain the same)
    st.title("🔍 S3 Spark SQL Query Tool")

    # --- LOAD DATA SECTION ---
    with st.expander("📂 Load Data", expanded=not st.session_state['tables']):
        load_method = st.radio("Load from:", ("S3", "Local"), horizontal=True)

        if load_method == "S3":
            if not st.session_state['credentials']:
                st.warning(
                    "S3 credentials not set. Please configure them in the sidebar.")
            else:
                s3_path = st.text_input("S3 Path (s3://bucket/path):")
                file_format = st.selectbox("File Format:", ["parquet", "csv"])
                table_name = st.text_input(
                    "Table Name:", f"tbl_{int(time.time())}")

                csv_options = {}
                if file_format == 'csv':
                    with st.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            csv_options['header'] = st.toggle(
                                "Header", value=True)
                            csv_options['inferSchema'] = st.toggle(
                                "Infer Schema", value=True)
                        with col2:
                            csv_options['sep'] = st.text_input(
                                "Delimiter:", ",")

                load_mode = st.radio("Load Mode:", [
                                     "Direct (for large datasets, slower queries)", "Download (for smaller datasets, faster queries)"], horizontal=True)

                if st.button("🚀 Load Table"):
                    if s3_path and table_name:
                        with st.spinner(f"Loading {table_name} from S3..."):
                            if load_mode == "Download":
                                success, msg, local_files, total_size, local_path = download_from_s3_with_progress(
                                    s3_path, file_format, table_name, st.session_state['credentials'])
                                if success:
                                    success, df, error = load_from_local_files(
                                        local_files, file_format, csv_options, st.session_state['credentials'])
                                    source_info = f"S3 (Downloaded to {local_path})"
                                else:
                                    st.error(msg)
                                    st.stop()
                            else:  # Direct load
                                success, df, error = load_from_s3_direct(
                                    s3_path, file_format, csv_options, st.session_state['credentials'])
                                source_info = f"S3 Direct ({s3_path})"
                                local_files = []
                                total_size = 0  # Can't easily get total size for direct load

                            if success:
                                df.createOrReplaceTempView(table_name)
                                st.session_state['tables'][table_name] = {
                                    'dataframe': df,
                                    'source': source_info,
                                    's3_path': s3_path,
                                    'file_format': file_format,
                                    'csv_options': csv_options if file_format == 'csv' else {},
                                    'local_files': local_files,
                                    'local_path': os.path.dirname(local_files[0]) if local_files else None,
                                    'size_bytes': total_size,
                                    'row_count': df.count(),
                                    'col_count': len(df.columns),
                                    'schema': [{'name': f.name, 'type': str(f.dataType)} for f in df.schema.fields],
                                    'loaded': True
                                }
                                save_metadata()
                                st.success(
                                    f"Table '{table_name}' loaded successfully!")
                                st.rerun()
                            else:
                                st.error(f"Error loading table: {error}")
                    else:
                        st.error("Please provide S3 path and table name.")

        elif load_method == "Local":
            st.info("Local mode uses files already downloaded or available locally.")
            local_path = st.text_input("Local Folder Path:", DOWNLOADS_DIR)
            file_format = st.selectbox(
                "File Format:", ["parquet", "csv"], key="local_format")
            table_name = st.text_input(
                "Table Name:", f"tbl_local_{int(time.time())}")

            csv_options = {}
            if file_format == 'csv':
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        csv_options['header'] = st.toggle(
                            "Header", value=True, key="local_header")
                        csv_options['inferSchema'] = st.toggle(
                            "Infer Schema", value=True, key="local_infer")
                    with col2:
                        csv_options['sep'] = st.text_input(
                            "Delimiter:", ",", key="local_sep")

            if st.button("🚀 Load Local Table"):
                if local_path and table_name:
                    if os.path.exists(local_path) and os.path.isdir(local_path):
                        with st.spinner(f"Loading {table_name} from {local_path}..."):
                            local_files = [os.path.join(local_path, f) for f in os.listdir(
                                local_path) if not f.startswith('.')]
                            if not local_files:
                                st.error(f"No files found in {local_path}")
                                st.stop()

                            success, df, error = load_from_local_files(
                                local_files, file_format, csv_options, st.session_state['credentials'])

                            if success:
                                total_size = sum(os.path.getsize(f) for f in local_files)
                                df.createOrReplaceTempView(table_name)
                                st.session_state['tables'][table_name] = {
                                    'dataframe': df,
                                    'source': f"Local ({local_path})",
                                    's3_path': None,
                                    'file_format': file_format,
                                    'csv_options': csv_options if file_format == 'csv' else {},
                                    'local_files': local_files,
                                    'local_path': local_path,
                                    'size_bytes': total_size,
                                    'row_count': df.count(),
                                    'col_count': len(df.columns),
                                    'schema': [{'name': f.name, 'type': str(f.dataType)} for f in df.schema.fields],
                                    'loaded': True
                                }
                                save_metadata()
                                st.success(
                                    f"Table '{table_name}' loaded successfully from local files!")
                                st.rerun()
                            else:
                                st.error(
                                    f"Error loading local table: {error}")
                    else:
                        st.error(f"Local path '{local_path}' does not exist or is not a directory.")
                else:
                    st.error("Please provide a local path and table name.")

    # --- SQL QUERY SECTION ---
    if st.session_state['tables']:
        st.markdown("### 📝 SQL Query")

        # If a query was staged by the AI, use it, otherwise use session state
        if st.session_state.get('staged_query'):
            query_text = st.session_state.get('staged_query')
            st.session_state['staged_query'] = None  # Clear after use
        else:
            query_text = st.session_state.get('sql_query', 'SELECT * FROM ... LIMIT 100')


        query = st.text_area("Enter your SQL query here:",
                             value=query_text, height=150)
        st.session_state['sql_query'] = query # Save user input

        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("▶️ Run Query", use_container_width=True):
                if query.strip():
                    with st.spinner("Executing query..."):
                        try:
                            spark = get_spark_session(
                                st.session_state['credentials'])
                            start_time = time.time()
                            result_df = spark.sql(query)
                            st.session_state['last_query_result_df'] = result_df
                            st.session_state['last_query_time'] = time.time() - start_time
                            st.session_state['last_query_error'] = None
                            st.rerun() # Rerun to display results
                        except Exception as e:
                            st.session_state['last_query_result_df'] = None
                            st.session_state['last_query_error'] = str(e)
                            st.rerun()
                else:
                    st.warning("Query is empty.")
        
        with col2:
             if 'last_query_result_df' in st.session_state and st.session_state['last_query_result_df'] is not None:
                df_to_download = st.session_state['last_query_result_df']
                csv_data = dataframe_to_csv_string(df_to_download)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"query_result_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True
                )


        if 'last_query_error' in st.session_state and st.session_state['last_query_error']:
            st.error(f"Query Error: {st.session_state['last_query_error']}")

        if 'last_query_result_df' in st.session_state and st.session_state['last_query_result_df'] is not None:
            result_df = st.session_state['last_query_result_df']
            query_time = st.session_state['last_query_time']
            st.success(
                f"Query executed in {query_time:.2f} seconds. Displaying top 500 rows.")

            # Use st.dataframe for better display
            st.dataframe(result_df.limit(500).toPandas(), height=400)

            with st.expander("Result Schema"):
                st.json([{'name': f.name, 'type': str(f.dataType)} for f in result_df.schema.fields])


    # --- AI DATA ANALYSIS SECTION ---
    if st.session_state['openai_key'] and st.session_state['tables']:
        st.markdown(f"### {ui_cfg.get('ai_section_title')}")

        with st.container():
            table_to_analyze = st.selectbox(
                "Select a table to analyze:", list(st.session_state['tables'].keys()))
            user_question = st.text_input(
                "Ask a question about the data:", placeholder="e.g., What's the average value of 'column_x'?")

            if st.button("🧠 Analyze with AI"):
                if table_to_analyze and user_question:
                    with st.spinner("AI is thinking..."):
                        response_placeholder = st.empty()
                        full_response = ""
                        try:
                            stream = analyze_data_with_ai(table_to_analyze, user_question,
                                                          st.session_state['credentials'], st.session_state['openai_key'], stream=True)
                            for chunk in stream:
                                if "error" in chunk:
                                    st.error(f"AI Error: {chunk['error']}")
                                    break
                                content = chunk.choices[0].delta.content or ""
                                full_response += content
                                response_placeholder.markdown(
                                    full_response + "▌")
                            response_placeholder.markdown(full_response)

                            # Save conversation
                            st.session_state['ai_conversation'].insert(0, {
                                'type': 'question',
                                'table': table_to_analyze,
                                'question': user_question,
                                'response': full_response
                            })

                            # Check for SQL in response and offer to run it
                            sql_match = re.search(
                                r"```sql\n(.*?)```", full_response, re.DOTALL)
                            if sql_match:
                                sql_query = sql_match.group(1).strip()
                                if st.button("▶️ Run Extracted SQL Query"):
                                    st.session_state['staged_query'] = sql_query
                                    st.rerun()

                        except Exception as e:
                            st.error(f"An unexpected error occurred: {e}")
                else:
                    st.warning("Please select a table and ask a question.")

        # Display conversation history
        if st.session_state['ai_conversation']:
            st.markdown("---")
            st.markdown("#### History")
            for i, entry in enumerate(st.session_state['ai_conversation']):
                _display_conversation_entry(len(st.session_state['ai_conversation']) - i, entry)




if __name__ == "__main__":
    main()
