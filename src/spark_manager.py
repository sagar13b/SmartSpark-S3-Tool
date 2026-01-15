import os
import boto3
from botocore.exceptions import ClientError
from pyspark.sql import SparkSession
import streamlit as st
from .utils import (
    DOWNLOADS_DIR, 
    TEMP_DATA_DIR, 
    format_bytes,
    load_app_config
)

@st.cache_resource
def get_spark_session(credentials):
    """
    Creates and configures a SparkSession with cloud credentials.
    Uses st.cache_resource to ensure the SparkSession is created only once.
    """
    # Load config inside the function to avoid circular imports or early loading
    ui_cfg = load_app_config()
    
    builder = SparkSession.builder.appName(ui_cfg.get("app_title")) \
        .config("spark.local.dir", os.path.join(os.getcwd(), TEMP_DATA_DIR, "spark"))

    if credentials:
        if credentials.get('type') == 'aws':
            builder = builder \
                .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
                .config("spark.hadoop.fs.s3a.access.key", credentials.get('access_key', '')) \
                .config("spark.hadoop.fs.s3a.secret.key", credentials.get('secret_key', '')) \
                .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

            if credentials.get('session_token'):
                builder = builder.config("spark.hadoop.fs.s3a.session.token", credentials.get('session_token'))
        elif credentials.get('type') == 'azure':
            builder = builder \
                .config("spark.jars.packages", "org.apache.hadoop:hadoop-azure:3.3.4,com.microsoft.azure:azure-storage:8.6.6") \
                .config(f"spark.hadoop.fs.azure.account.key.{credentials.get('account_name')}.blob.core.windows.net", credentials.get('account_key', ''))

    return builder \
        .config("spark.ui.enabled", "true") \
        .config("spark.ui.port", "4040") \
        .getOrCreate()


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

            progress = downloaded_size / total_size if total_size > 0 else 0
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

        if 'tables' not in st.session_state:
            st.session_state['tables'] = {}

        st.session_state['tables'][table_name] = {
            'dataframe': df,
            'temp_dir': None,
            'loaded': True,
            **{k: v for k, v in meta.items()}
        }

        return True, f"Restored from {source_type}"
    except Exception as e:
        return False, str(e)


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
