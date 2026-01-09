import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
import boto3
from botocore.exceptions import ClientError
import tempfile
import os
import shutil
import json
from datetime import datetime
from pathlib import Path
import psutil
import time
import re
from openai import OpenAI


# --- CONFIGURATION & PATHS ---
STORAGE_FILE = "spark_tables_metadata.json"
CREDENTIALS_FILE = "aws_credentials.json"
OPENAI_KEY_FILE = "openai_key.json"
PROMPTS_FILE = "prompts.json"
DOWNLOADS_DIR = "spark_downloads"
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# Configuration files
STREAMLIT_CONFIG_DIR = Path.home() / ".streamlit"
STREAMLIT_CONFIG_FILE = STREAMLIT_CONFIG_DIR / "config.toml"


def save_openai_key(api_key):
    """Save OpenAI API key to file"""
    try:
        with open(OPENAI_KEY_FILE, 'w') as f:
            json.dump({'api_key': api_key}, f)
        return True
    except Exception as e:
        st.error(f"Error saving OpenAI key: {e}")
        return False


def load_openai_key():
    """Load OpenAI API key from file"""
    try:
        if os.path.exists(OPENAI_KEY_FILE):
            with open(OPENAI_KEY_FILE) as f:
                data = json.load(f)
                return data.get('api_key')
    except:
        pass
    return None


def validate_openai_key(api_key):
    """Validate OpenAI API key"""
    try:
        client = OpenAI(api_key=api_key)
        # Test with a simple request
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        return True, "API key is valid!"
    except Exception as e:
        return False, f"Invalid API key: {str(e)}"


def get_table_summary(table_name, credentials):
    """Get comprehensive summary of table for AI context"""
    try:
        table_info = st.session_state['tables'][table_name]
        spark = get_spark_session(credentials)
        df = table_info['dataframe']

        # Get sample data
        sample_data = df.limit(5).toPandas()

        # Get column statistics
        numeric_cols = [f.name for f in df.schema.fields if str(f.dataType) in [
            'IntegerType', 'LongType', 'FloatType', 'DoubleType', 'DecimalType']]
        stats_summary = ""

        if numeric_cols:
            stats_df = df.select(numeric_cols).summary(
                "count", "mean", "stddev", "min", "max").toPandas()
            stats_summary = stats_df.to_string()

        # Build summary
        summary = f"""
TABLE: {table_name}
Total Rows: {table_info['row_count']:,}
Total Columns: {table_info['col_count']}

SCHEMA:
{chr(10).join([f"- {col['name']}: {col['type']}" for col in table_info['schema']])}

SAMPLE DATA (first 5 rows):
{sample_data.to_string()}

NUMERIC COLUMN STATISTICS:
{stats_summary if stats_summary else "No numeric columns"}
"""
        return summary
    except Exception as e:
        return f"Error getting table summary: {str(e)}"


def execute_query_for_ai(query, credentials):
    """Execute a SQL query and return results as string"""
    try:
        spark = get_spark_session(credentials)
        result = spark.sql(query)
        pdf = result.limit(100).toPandas()
        return True, pdf.to_string(), len(pdf)
    except Exception as e:
        return False, str(e), 0


# --- PROMPT ENGINE ---
def load_prompts():
    """Load AI prompts from external JSON file with fallback defaults"""
    default_prompts = {
        "system_message": "You are a professional Data Analyst.",
        "one_shot_example": "",
        "analysis_instruction": "Analyze this table: {table_name}. Question: {user_question}",
        "results_interpretation": "Interpret these results: {query_result}"
    }
    try:
        if os.path.exists(PROMPTS_FILE):
            with open(PROMPTS_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading prompts.json: {e}")
    return default_prompts

# --- AI LOGIC (REFACTORED) ---
def analyze_data_with_ai(table_name, user_question, credentials, api_key, stream=False):
    """Analyze data using OpenAI with externalized prompts"""
    try:
        client = OpenAI(api_key=api_key)
        prompts = load_prompts()
        table_summary = get_table_summary(table_name, credentials)

        # Build the system context
        system_content = f"{prompts['system_message']}\n\nDATA CONTEXT:\n{table_summary}"
        
        # Inject One-Shot Example if it exists
        if prompts.get("one_shot_example"):
            system_content += f"\n\n{prompts['one_shot_example']}"

        # Build user prompt
        user_content = prompts["analysis_instruction"].format(
            table_name=table_name, 
            user_question=user_question
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=stream,
            temperature=0.7
        )
        return response
    except Exception as e:
        return f"Error: {str(e)}"


def analyze_with_query_results(table_name, user_question, query_result, credentials, api_key):
    """Follow-up analysis using results interpretation prompt"""
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


def save_credentials(credentials):
    """Save AWS credentials to file"""
    try:
        with open(CREDENTIALS_FILE, 'w') as f:
            json.dump(credentials, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving credentials: {e}")
        return False


def load_credentials():
    """Load AWS credentials from file"""
    try:
        if os.path.exists(CREDENTIALS_FILE):
            with open(CREDENTIALS_FILE) as f:
                return json.load(f)
    except:
        pass
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


def apply_theme(theme_mode):
    """Apply theme by modifying Streamlit config"""
    STREAMLIT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_content = f"""[theme]
base = "{theme_mode}"
"""
    with open(STREAMLIT_CONFIG_FILE, 'w') as f:
        f.write(config_content)


@st.cache_resource
def get_spark_session(credentials):
    """Create Spark session with AWS credentials"""
    builder = SparkSession.builder.appName("S3 Query Tool")

    if credentials:
        builder = builder \
            .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
            .config("spark.hadoop.fs.s3a.access.key", credentials.get('access_key', '')) \
            .config("spark.hadoop.fs.s3a.secret.key", credentials.get('secret_key', '')) \
            .config("spark.hadoop.fs.s3a.session.token", credentials.get('session_token', '')) \
            .config("spark.hadoop.fs.s3a.endpoint", "s3.us-east-1.amazonaws.com") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    return builder \
        .config("spark.ui.enabled", "true") \
        .config("spark.ui.port", "4040") \
        .getOrCreate()


def get_system_resources():
    """Get detailed system memory and disk usage"""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    process = psutil.Process(os.getpid())
    app_mem = process.memory_info().rss

    app_disk_usage = 0
    if os.path.exists(DOWNLOADS_DIR):
        for dirpath, dirnames, filenames in os.walk(DOWNLOADS_DIR):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    app_disk_usage += os.path.getsize(filepath)
                except:
                    pass

    return {
        'total_mem': mem.total,
        'used_mem': mem.used,
        'app_mem': app_mem,
        'total_disk': disk.total,
        'used_disk': disk.used,
        'app_disk': app_disk_usage,
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
    """Show OpenAI setup dialog"""
    st.title("🤖 AI Analysis Setup")
    st.markdown("### Enable AI-Powered Data Analysis")
    st.info("Enter your OpenAI API key to enable AI-powered data analysis features.")

    api_key = st.text_input(
        "OpenAI API Key:", type="password", placeholder="sk-...")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("💾 Save & Validate", type="primary"):
            if api_key.strip():
                with st.spinner("Validating API key..."):
                    is_valid, message = validate_openai_key(api_key)
                if is_valid:
                    if save_openai_key(api_key):
                        st.session_state['openai_key'] = api_key
                        st.session_state['show_openai_setup'] = False
                        st.success(f"✅ {message}")
                        time.sleep(1)
                        st.rerun()
                else:
                    st.error(f"❌ {message}")
            else:
                st.error("Please enter an API key")

    with col2:
        if st.button("⏭️ Skip (Disable AI Features)"):
            st.session_state['openai_key'] = None
            st.session_state['show_openai_setup'] = False
            st.rerun()


def show_credentials_setup():
    """Show credentials setup interface"""
    st.title("🔐 AWS Credentials Setup")
    st.markdown("""
    ### Welcome to S3 Spark SQL Query Tool!
    
    To use S3 features, please provide your AWS credentials. You can paste them in the export format:
    
    ```bash
    export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY"
    export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"
    export AWS_SESSION_TOKEN="YOUR_SESSION_TOKEN"  # Optional
    ```
    """)

    credentials_text = st.text_area(
        "Paste your AWS credentials here:",
        height=200,
        placeholder="""export AWS_ACCESS_KEY_ID="ASIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."  # Optional for temporary credentials"""
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("💾 Save & Validate", type="primary"):
            if credentials_text.strip():
                credentials = parse_aws_credentials(credentials_text)
                if credentials:
                    with st.spinner("Validating credentials..."):
                        is_valid, message = validate_s3_credentials(
                            credentials)
                    if is_valid:
                        if save_credentials(credentials):
                            st.session_state['credentials'] = credentials
                            st.session_state['credentials_validated'] = True
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error("Failed to save credentials")
                    else:
                        st.error(f"❌ {message}")
                else:
                    st.error(
                        "Could not parse credentials. Please check the format.")
            else:
                st.error("Please enter your credentials")

    with col2:
        if st.button("⏭️ Skip (Local Files Only)"):
            st.session_state['credentials'] = None
            st.session_state['credentials_validated'] = False
            st.rerun()


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
        st.session_state['credentials'] = load_credentials()
    if 'credentials_validated' not in st.session_state:
        st.session_state['credentials_validated'] = st.session_state['credentials'] is not None
    if 'openai_key' not in st.session_state:
        st.session_state['openai_key'] = load_openai_key()
    if 'ai_conversation' not in st.session_state:
        st.session_state['ai_conversation'] = []
    if 'show_openai_setup' not in st.session_state:
        st.session_state['show_openai_setup'] = False

    # Check if OpenAI setup is requested
    if st.session_state.get('show_openai_setup', False) and not st.session_state['openai_key']:
        show_openai_setup()
        return

    # Check if credentials are set up
    if st.session_state['credentials'] is None and 'skip_credentials' not in st.session_state:
        show_credentials_setup()
        return

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Theme toggle
        current_theme = st.session_state.get('theme', 'light')
        theme_label = "🌙 Dark" if current_theme == 'light' else "☀️ Light"
        if st.button(theme_label, use_container_width=True):
            new_theme = 'dark' if current_theme == 'light' else 'light'
            st.session_state['theme'] = new_theme
            apply_theme(new_theme)
            st.rerun()

        # Credentials status
        if st.session_state['credentials']:
            st.success("✅ AWS: Connected")
            if st.button("🔄 Update Credentials", use_container_width=True):
                st.session_state['credentials'] = None
                st.session_state['credentials_validated'] = False
                st.rerun()
        else:
            st.warning("⚠️ AWS: Not configured (Local only)")
            if st.button("🔐 Setup AWS", use_container_width=True):
                st.session_state['credentials'] = None
                st.rerun()

        # OpenAI status
        if st.session_state['openai_key']:
            st.success("✅ AI: Enabled")
            if st.button("🔄 Update OpenAI Key", use_container_width=True):
                st.session_state['openai_key'] = None
                st.session_state['show_openai_setup'] = True
                st.rerun()
        else:
            st.warning("⚠️ AI: Disabled")
            if st.button("🤖 Setup OpenAI", use_container_width=True):
                st.session_state['show_openai_setup'] = True
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
                    status = "S3 Only"

                with st.expander(f"{status_icon} {table_name} ({status})"):
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
                                        "AWS credentials required for S3 tables")
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

        # Active Tables Management
        if st.session_state['tables']:
            st.markdown("### 📊 Active Tables")
            for name, info in st.session_state['tables'].items():
                with st.expander(f"✅ {name}"):
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

        if st.button("💾 Save Metadata", use_container_width=True):
            if save_metadata():
                st.session_state['metadata_cache'] = load_metadata()
                st.success("✅ Saved!")

        if st.button("🗑️ Clear All Active", use_container_width=True):
            st.session_state['tables'] = {}
            st.rerun()

    # Main area
    st.title("🔍 S3 Spark SQL Query Tool")

    # AI Analysis Section (if OpenAI is set up and tables exist)
    if st.session_state['openai_key'] and st.session_state['tables']:
        st.subheader("🤖 AI Data Analysis")

        selected_table = st.selectbox(
            "Select table for AI analysis:",
            options=list(st.session_state['tables'].keys()),
            key="ai_table_select"
        )

        user_question = st.text_area(
            "Ask a question about your data:",
            placeholder="Examples:\n- What are the top 10 customers by revenue?\n- Show me sales trends over time\n- Find anomalies in the data\n- What's the average transaction value by region?",
            height=100,
            key="ai_question"
        )

        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            analyze_button = st.button("🔍 Analyze", type="primary")
        with col2:
            if st.button("🗑️ Clear Conversation"):
                st.session_state['ai_conversation'] = []
                st.rerun()

        if analyze_button and user_question.strip():
            with st.spinner("🤔 AI is analyzing your data..."):
                # Get initial analysis
                response_stream = analyze_data_with_ai(
                    selected_table,
                    user_question,
                    st.session_state['credentials'],
                    st.session_state['openai_key'],
                    stream=True
                )

                # Display streaming response
                response_placeholder = st.empty()
                full_response = ""

                for chunk in response_stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

                # Store in conversation
                st.session_state['ai_conversation'].append({
                    'question': user_question,
                    'response': full_response,
                    'table': selected_table
                })

                # Check if response contains SQL query
                sql_match = re.search(
                    r'```sql\n(.*?)\n```', full_response, re.DOTALL)

                if sql_match:
                    suggested_query = sql_match.group(1).strip()
                    st.info("💡 AI suggested a SQL query to get more insights!")
                    st.code(suggested_query, language="sql")

                    if st.button("▶️ Run Suggested Query", key="run_ai_query"):
                        try:
                            with st.spinner("Executing query..."):
                                success, result_str, row_count = execute_query_for_ai(
                                    suggested_query,
                                    st.session_state['credentials']
                                )

                                if success:
                                    st.success(
                                        f"✅ Query executed! Returned {row_count} rows")

                                    # Show results
                                    spark = get_spark_session(
                                        st.session_state['credentials'])
                                    result_df = spark.sql(suggested_query)
                                    pdf = result_df.limit(100).toPandas()
                                    st.dataframe(pdf, use_container_width=True)

                                    # Get AI analysis of results
                                    with st.spinner("🤔 AI is analyzing the results..."):
                                        follow_up = analyze_with_query_results(
                                            selected_table,
                                            user_question,
                                            result_str,
                                            st.session_state['credentials'],
                                            st.session_state['openai_key']
                                        )

                                        st.markdown("### 📊 Analysis:")
                                        st.markdown(follow_up)

                                        # Store follow-up in conversation
                                        st.session_state['ai_conversation'].append({
                                            'query': suggested_query,
                                            'results': result_str,
                                            'analysis': follow_up,
                                            'table': selected_table
                                        })
                                else:
                                    st.error(f"Query failed: {result_str}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

        # Show conversation history
        if st.session_state['ai_conversation']:
            with st.expander("📜 Conversation History", expanded=False):
                for idx, conv in enumerate(reversed(st.session_state['ai_conversation']), 1):
                    conv_num = len(
                        st.session_state['ai_conversation']) - idx + 1

                    if 'question' in conv:
                        with st.container():
                            col1, col2 = st.columns([5, 1])
                            with col1:
                                st.markdown(
                                    f"**#{conv_num} - Table: {conv['table']}**")
                            with col2:
                                expand_key = f"expand_conv_{conv_num}"
                                if expand_key not in st.session_state:
                                    st.session_state[expand_key] = False

                                if st.button("🔍" if not st.session_state[expand_key] else "🔼",
                                             key=f"btn_expand_{conv_num}",
                                             help="Expand/Collapse"):
                                    st.session_state[expand_key] = not st.session_state[expand_key]
                                    st.rerun()

                            st.markdown(f"**Q:** {conv['question']}")

                            if st.session_state.get(expand_key, False):
                                st.markdown("**A:**")
                                st.markdown(conv['response'])
                            else:
                                response_preview = conv['response'][:300] + "..." if len(
                                    conv['response']) > 300 else conv['response']
                                st.markdown(f"**A:** {response_preview}")

                            st.markdown("---")

                    elif 'query' in conv:
                        with st.container():
                            col1, col2 = st.columns([5, 1])
                            with col1:
                                st.markdown(
                                    f"**#{conv_num} - Query Results Analysis**")
                            with col2:
                                expand_key = f"expand_conv_{conv_num}"
                                if expand_key not in st.session_state:
                                    st.session_state[expand_key] = False

                                if st.button("🔍" if not st.session_state[expand_key] else "🔼",
                                             key=f"btn_expand_{conv_num}",
                                             help="Expand/Collapse"):
                                    st.session_state[expand_key] = not st.session_state[expand_key]
                                    st.rerun()

                            st.code(conv['query'], language="sql")

                            if st.session_state.get(expand_key, False):
                                st.markdown("**Analysis:**")
                                st.markdown(conv['analysis'])
                            else:
                                analysis_preview = conv['analysis'][:300] + "..." if len(
                                    conv['analysis']) > 300 else conv['analysis']
                                st.markdown(
                                    f"**Analysis:** {analysis_preview}")

                            st.markdown("---")

                    elif 'type' in conv and conv['type'] == 'sql_analysis':
                        with st.container():
                            col1, col2 = st.columns([5, 1])
                            with col1:
                                st.markdown(
                                    f"**#{conv_num} - SQL Results Analysis**")
                            with col2:
                                expand_key = f"expand_conv_{conv_num}"
                                if expand_key not in st.session_state:
                                    st.session_state[expand_key] = False

                                if st.button("🔍" if not st.session_state[expand_key] else "🔼",
                                             key=f"btn_expand_{conv_num}",
                                             help="Expand/Collapse"):
                                    st.session_state[expand_key] = not st.session_state[expand_key]
                                    st.rerun()

                            st.code(conv['query'], language="sql")

                            if st.session_state.get(expand_key, False):
                                st.markdown("**Full Analysis:**")
                                st.markdown(conv['analysis'])
                            else:
                                analysis_preview = conv['analysis'][:300] + "..." if len(
                                    conv['analysis']) > 300 else conv['analysis']
                                st.markdown(
                                    f"**Analysis:** {analysis_preview}")

                            st.markdown("---")

        st.markdown("---")

    # Check if OpenAI needs setup
    elif not st.session_state['openai_key'] and st.session_state['tables']:
        with st.expander("🤖 Enable AI Analysis", expanded=False):
            show_openai_setup()

    # SQL Query section
    if st.session_state['tables']:
        st.subheader("🔍 SQL Query")

        tables_list = list(st.session_state['tables'].keys())

        with st.expander("📋 Available Tables & Columns", expanded=False):
            for table_name in tables_list:
                st.markdown(f"**{table_name}:**")
                cols = st.session_state['tables'][table_name].get(
                    'columns', [])
                st.code(", ".join(cols), language="text")

        default_query = f"SELECT * FROM {tables_list[0]} LIMIT 100" if tables_list else ""
        query = st.text_area("Enter SQL query:",
                             value=default_query, height=150)

        if st.button("▶️ Execute Query", type="primary"):
            start_time = time.time()
            with st.spinner("Executing query..."):
                spark = get_spark_session(st.session_state['credentials'])
                result = spark.sql(query)

                pdf = result.toPandas()
                execution_time = time.time() - start_time

                data_scanned = sum([st.session_state['tables'][t].get('size_bytes', 0)
                                   for t in tables_list if t in query])

                # Store results in session state for AI analysis
                st.session_state['last_query_results'] = {
                    'query': query,
                    'dataframe': pdf,
                    'execution_time': execution_time,
                    'data_scanned': data_scanned,
                    'execution_plan': result._jdf.queryExecution().simpleString()
                }
                st.rerun()

        # Display results if available
        if st.session_state.get('last_query_results'):
            last_results = st.session_state['last_query_results']
            pdf = last_results['dataframe']
            execution_time = last_results['execution_time']
            data_scanned = last_results['data_scanned']

            with st.expander("🔍 Query Execution Plan"):
                st.text(last_results['execution_plan'])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows Returned", f"{len(pdf):,}")
            with col2:
                st.metric("Execution Time", f"{execution_time:.2f}s")
            with col3:
                st.metric("Data Scanned (approx)", format_bytes(data_scanned))

            st.dataframe(pdf, use_container_width=True)

            csv = pdf.to_csv(index=False)

            # AI Analysis of Query Results (only show if AI is enabled)
            if st.session_state.get('openai_key'):
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.download_button("📥 Download CSV", csv,
                                       f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                       key="download_csv_main")
                with col2:
                    if st.button("🤖 Analyze Results with AI", key="analyze_sql_results"):
                        st.session_state['trigger_ai_analysis'] = True
                        st.rerun()
            else:
                st.download_button("📥 Download CSV", csv,
                                   f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                   key="download_csv_main")

        # Show AI analysis if triggered
        if st.session_state.get('trigger_ai_analysis', False) and st.session_state.get('last_query_results'):
            st.session_state['trigger_ai_analysis'] = False

            last_results = st.session_state['last_query_results']
            pdf = last_results['dataframe']
            query = last_results['query']

            st.markdown("---")

            with st.spinner("🤔 AI is analyzing the query results..."):
                # Prepare results for AI
                result_summary = f"""
Query executed: {query}

Results Summary:
- Rows returned: {len(pdf):,}
- Execution time: {last_results['execution_time']:.2f}s

Sample of results (first 20 rows):
{pdf.head(20).to_string()}

Full statistics:
{pdf.describe().to_string() if len(pdf.describe().columns) > 0 else 'No numeric columns'}
"""

                # Get AI analysis
                try:
                    client = OpenAI(api_key=st.session_state['openai_key'])

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a data analyst expert. Analyze the SQL query results and provide insights."},
                            {"role": "user", "content": f"""Analyze these query results and provide:

{result_summary}

Please provide:
1. **Key Findings**: What are the main insights from this data?
2. **Patterns & Trends**: Any notable patterns, trends, or anomalies?
3. **Data Quality**: Any concerns about data quality or completeness?
4. **Recommendations**: What actions or further analysis would you recommend?
5. **Follow-up Questions**: Suggest 2-3 follow-up queries that would provide additional insights.

Format your response with clear sections using markdown."""}
                        ],
                        temperature=0.7,
                        max_tokens=2000
                    )

                    analysis = response.choices[0].message.content

                    st.markdown("---")
                    st.markdown("### 🤖 AI Analysis of Query Results")
                    st.markdown(analysis)

                    # Store in conversation
                    st.session_state['ai_conversation'].append({
                        'type': 'sql_analysis',
                        'query': query,
                        'results_summary': result_summary,
                        'analysis': analysis,
                        'timestamp': datetime.now().isoformat()
                    })

                except Exception as e:
                    st.error(f"Error analyzing results: {str(e)}")

        st.markdown("---")

    # Load data section
    st.subheader("📥 Load Data")

    source = st.radio("Data Source:", ["S3", "Local Files"], horizontal=True)

    # Disable S3 if no credentials
    if source == "S3" and not st.session_state.get('credentials'):
        st.warning(
            "⚠️ AWS credentials required for S3 features. Click 'Setup AWS' in sidebar.")
        source = "Local Files"

    col1, col2 = st.columns([2, 1])
    with col1:
        table_name = st.text_input("Table Name:", placeholder="my_table")
    with col2:
        file_format = st.selectbox("Format:", ["parquet", "csv"])

    csv_options = {}
    if file_format == "csv":
        with st.expander("CSV Options"):
            col1, col2 = st.columns(2)
            with col1:
                csv_options['header'] = st.checkbox("Header", value=True)
            with col2:
                csv_options['inferSchema'] = st.checkbox(
                    "Infer Schema", value=True)

    if source == "S3":
        s3_path = st.text_input("S3 Path:", placeholder="s3://bucket/path/")
        load_method = st.radio("Load Method:", [
                               "Direct Load (Faster)", "Download and Cache Locally"], horizontal=True)

        if st.button("📥 Load from S3", type="primary"):
            if not s3_path or not table_name:
                st.error("Enter both S3 path and table name")
            elif table_name in st.session_state['tables']:
                st.error(f"Table '{table_name}' already loaded")
            else:
                if load_method == "Direct Load (Faster)":
                    with st.spinner("Loading directly from S3..."):
                        success, df, error = load_from_s3_direct(
                            s3_path, file_format, csv_options, st.session_state['credentials'])
                        if success:
                            df.cache()
                            df.createOrReplaceTempView(table_name)

                            st.session_state['tables'][table_name] = {
                                'dataframe': df,
                                'temp_dir': None,
                                'file_count': 1,
                                'row_count': df.count(),
                                'col_count': len(df.columns),
                                'source': f'S3: {s3_path}',
                                'size_bytes': 0,
                                'columns': df.columns,
                                'schema': [{'name': f.name, 'type': str(f.dataType)} for f in df.schema.fields],
                                'created_at': datetime.now().isoformat(),
                                's3_path': s3_path,
                                'file_format': file_format,
                                'csv_options': csv_options,
                                'loaded': True,
                                'local_files': [],
                                'local_path': ''
                            }

                            save_metadata()
                            st.session_state['metadata_cache'] = load_metadata(
                            )
                            st.success(f"✅ Loaded {table_name}!")
                            st.dataframe(df.limit(10).toPandas(),
                                         use_container_width=True)
                            st.rerun()
                        else:
                            st.error(f"Error: {error}")
                else:
                    success, message, files, total_size, local_path = download_from_s3_with_progress(
                        s3_path, file_format, table_name, st.session_state['credentials'])

                    if success:
                        with st.spinner("Loading into Spark..."):
                            spark_success, df, error = load_from_local_files(
                                files, file_format, csv_options, st.session_state['credentials'])

                            if spark_success:
                                df.cache()
                                df.createOrReplaceTempView(table_name)

                                st.session_state['tables'][table_name] = {
                                    'dataframe': df,
                                    'temp_dir': None,
                                    'file_count': len(files),
                                    'row_count': df.count(),
                                    'col_count': len(df.columns),
                                    'source': f'S3: {s3_path}',
                                    'size_bytes': total_size,
                                    'columns': df.columns,
                                    'schema': [{'name': f.name, 'type': str(f.dataType)} for f in df.schema.fields],
                                    'created_at': datetime.now().isoformat(),
                                    's3_path': s3_path,
                                    'file_format': file_format,
                                    'csv_options': csv_options,
                                    'loaded': True,
                                    'local_files': files,
                                    'local_path': local_path
                                }

                                save_metadata()
                                st.session_state['metadata_cache'] = load_metadata(
                                )
                                st.success(
                                    f"✅ Loaded {table_name}! Cached locally: {format_bytes(total_size)}")
                                st.dataframe(df.limit(10).toPandas(),
                                             use_container_width=True)
                                st.rerun()
                            else:
                                st.error(f"Error loading into Spark: {error}")
                    else:
                        st.error(message)

    else:  # Local files
        uploaded = st.file_uploader(
            f"Upload {file_format.upper()} files:",
            type=['parquet', 'parq'] if file_format == 'parquet' else [
                'csv', 'txt'],
            accept_multiple_files=True
        )

        if st.button("📥 Load Local Files", type="primary") and uploaded:
            if not table_name:
                st.error("Enter table name")
            elif table_name in st.session_state['tables']:
                st.error(f"Table '{table_name}' already loaded")
            else:
                try:
                    temp_dir = tempfile.mkdtemp()
                    files = []
                    total_size = 0

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for idx, f in enumerate(uploaded, 1):
                        path = os.path.join(temp_dir, f.name)
                        with open(path, 'wb') as out:
                            out.write(f.getbuffer())
                        files.append(path)
                        total_size += f.size

                        progress_bar.progress(idx / len(uploaded))
                        status_text.text(
                            f"Uploading: {format_bytes(total_size)} ({idx}/{len(uploaded)} files)")

                    progress_bar.empty()
                    status_text.empty()

                    with st.spinner("Loading into Spark..."):
                        spark = get_spark_session(
                            st.session_state['credentials'])
                        if file_format == "parquet":
                            df = spark.read.parquet(*files)
                        else:
                            reader = spark.read.format("csv")
                            for k, v in csv_options.items():
                                reader = reader.option(k, v)
                            df = reader.load(*files)

                        df.cache()
                        df.createOrReplaceTempView(table_name)

                        st.session_state['tables'][table_name] = {
                            'dataframe': df,
                            'temp_dir': temp_dir,
                            'file_count': len(files),
                            'row_count': df.count(),
                            'col_count': len(df.columns),
                            'source': f'Local: {len(files)} file(s)',
                            'size_bytes': total_size,
                            'columns': df.columns,
                            'schema': [{'name': f.name, 'type': str(f.dataType)} for f in df.schema.fields],
                            'created_at': datetime.now().isoformat(),
                            's3_path': '',
                            'file_format': file_format,
                            'csv_options': csv_options,
                            'loaded': True,
                            'local_files': files,
                            'local_path': temp_dir
                        }

                        save_metadata()
                        st.session_state['metadata_cache'] = load_metadata()
                        st.success(f"✅ Loaded {table_name}!")
                        st.dataframe(df.limit(10).toPandas(),
                                     use_container_width=True)
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # System Resources at bottom (manual refresh only)
    st.markdown("---")
    with st.expander("💻 System Resources", expanded=False):
        if st.button("🔄 Refresh Resources"):
            resources = get_system_resources()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📊 Memory (RAM)**")
                mem_pct = (resources['used_mem'] /
                           resources['total_mem']) * 100
                st.progress(mem_pct / 100)
                st.caption(
                    f"Used: {format_bytes(resources['used_mem'])} / {format_bytes(resources['total_mem'])} ({mem_pct:.1f}%)")
                st.caption(
                    f"└─ This App: {format_bytes(resources['app_mem'])}")

            with col2:
                st.markdown("**💾 Disk Storage**")
                disk_pct = (resources['used_disk'] /
                            resources['total_disk']) * 100
                st.progress(disk_pct / 100)
                st.caption(
                    f"Used: {format_bytes(resources['used_disk'])} / {format_bytes(resources['total_disk'])} ({disk_pct:.1f}%)")
                st.caption(
                    f"└─ This App: {format_bytes(resources['app_disk'])}")


if __name__ == "__main__":
    main()
