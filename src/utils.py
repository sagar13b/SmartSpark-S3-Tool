import os
import json
import shutil
from pathlib import Path

import streamlit as st

# --- CONFIGURATION & PATHS ---
CONFIG_FILE = "config.json"
STORAGE_FILE = "spark_tables_metadata.json"
AWS_CREDENTIALS_ENV_PREFIX = "AWS_"
AZURE_CREDENTIALS_ENV_PREFIX = "AZURE_"
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
PROMPTS_FILE = "prompts.json"

# Temporary directories
TEMP_DATA_DIR = "temp_data"
DOWNLOADS_DIR = "spark_downloads"

# Ensure directories exist
os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(TEMP_DATA_DIR, exist_ok=True)

# Configuration files
STREAMLIT_CONFIG_DIR = ".streamlit"
os.makedirs(STREAMLIT_CONFIG_DIR, exist_ok=True)
STREAMLIT_CONFIG_FILE = Path(".streamlit/config.toml")


def update_env_file(key, value):
    """
    Updates a key-value pair in the .env file.
    Creates the .env file if it doesn't exist.
    """
    env_path = ".env"
    
    # Read existing lines
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()
    else:
        lines = []

    key_found = False
    new_lines = []
    
    # Update existing key
    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(f'{key}="{value}"\n')
            key_found = True
        else:
            new_lines.append(line)
    
    # Add new key if not found
    if not key_found:
        if new_lines and not new_lines[-1].endswith('\n'):
            new_lines[-1] += '\n'
        new_lines.append(f'{key}="{value}"\n')
    
    # Write back to file
    with open(env_path, "w") as f:
        f.writelines(new_lines)


def load_app_config():
    """Load UI settings like Title and Headers"""
    default_config = {
        "app_title": "SmartSpark S3 Explorer",
        "header_title": "🚀 SmartSpark Data Workbench",
        "sidebar_header": "Data Controls",
        "ai_section_title": "🧠 AI Data Analyst",
        "theme_color": "#FF4B4B"
    }
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                # Merge with defaults
                return {**default_config, **config}
        except:
            return default_config
    return default_config


def format_bytes(b):
    """Format bytes to human readable string"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024:
            return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} PB"


def save_metadata():
    """Save the current table metadata to disk"""
    if 'tables' not in st.session_state:
        return False
        
    metadata = {}
    for name, info in st.session_state['tables'].items():
        # Only save serializable info
        metadata[name] = {
            'source': info['source'],
            'row_count': info['row_count'],
            'col_count': info['col_count'],
            'local_files': info.get('local_files', []),
            's3_path': info.get('s3_path'),
            'file_format': info.get('file_format'),
            'csv_options': info.get('csv_options', {})
        }
    
    try:
        with open(STORAGE_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving metadata: {e}")
        return False


def load_metadata():
    """Load table metadata from disk"""
    if os.path.exists(STORAGE_FILE):
        try:
            with open(STORAGE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def delete_table_metadata(table_name):
    """Delete table from metadata and clean up downloaded files"""
    meta = load_metadata()
    
    if table_name in meta:
        table_info = meta[table_name]
        
        # 1. Delete local files if they exist
        local_files = table_info.get('local_files', [])
        for file_path in local_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
                
        # Try to remove the directory if it's in downloads and empty
        if local_files:
            dir_path = os.path.dirname(local_files[0])
            if "spark_downloads" in dir_path:
                try:
                    remaining_files = os.listdir(dir_path)
                    if not remaining_files or (len(remaining_files) == 1 and remaining_files[0] == '.DS_Store'):
                         shutil.rmtree(dir_path)
                except:
                    pass

        # 2. Remove from JSON
        del meta[table_name]
        
        try:
            with open(STORAGE_FILE, 'w') as f:
                json.dump(meta, f, indent=2)
            return True, f"Table '{table_name}' and associated files deleted."
        except Exception as e:
            return False, f"Error saving metadata: {e}"
            
    return False, "Table not found in metadata."
