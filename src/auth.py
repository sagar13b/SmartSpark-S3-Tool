import os
import re
import boto3
from botocore.exceptions import ClientError
from openai import OpenAI
import streamlit as st
from .utils import (
    AWS_CREDENTIALS_ENV_PREFIX,
    AZURE_CREDENTIALS_ENV_PREFIX,
    OPENAI_API_KEY_ENV_VAR
)

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
