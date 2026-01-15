import time
import os
import shutil
import re
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Import from new modules
from src.utils import (
    load_app_config,
    save_metadata,
    load_metadata,
    delete_table_metadata,
    format_bytes,
    TEMP_DATA_DIR,
    DOWNLOADS_DIR
)
from src.auth import (
    load_credentials, 
    load_openai_key, 
    validate_s3_credentials, 
    validate_azure_credentials
)
from src.ui_components import (
    apply_theme,
    show_openai_setup,
    show_credentials_setup,
    _display_conversation_entry
)
from src.spark_manager import (
    get_spark_session,
    download_from_s3_with_progress,
    load_from_s3_direct,
    load_from_local_files,
    restore_table,
    dataframe_to_csv_string
)
from src.ai_manager import analyze_data_with_ai
from src.system import get_system_resources

def main():
    ui_cfg = load_app_config()
    st.set_page_config(
        page_title=ui_cfg.get("app_title", "SmartSpark S3 Explorer"),
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
        
        valid_creds = None
        cloud_type = None

        if aws_creds:
            is_valid, msg = validate_s3_credentials(aws_creds)
            if is_valid:
                valid_creds = aws_creds
                cloud_type = 'aws'
        
        if not valid_creds and azure_creds:
             is_valid, msg = validate_azure_credentials(azure_creds)
             if is_valid:
                 valid_creds = azure_creds
                 cloud_type = 'azure'

        st.session_state['credentials'] = valid_creds
        st.session_state['cloud_type'] = cloud_type

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
                    cloud_help = "Connected"
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

    # Main area
    st.title(f"🔍 {ui_cfg.get('app_title', 'S3 Spark SQL Query Tool')}")

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
            local_source_mode = st.radio("Local Source:", ["Upload Files", "Existing Folder Path"], horizontal=True)
            
            local_path = None
            uploaded_files = []
            
            if local_source_mode == "Upload Files":
                uploaded_files = st.file_uploader(
                    "Choose files", 
                    accept_multiple_files=True,
                    type=['csv', 'parquet', 'txt']
                )
                if uploaded_files:
                     table_name_default = os.path.splitext(uploaded_files[0].name)[0]
                else:
                     table_name_default = f"tbl_local_{int(time.time())}"
            else:
                local_path_input = st.text_input("Local Folder Path:", DOWNLOADS_DIR)
                if local_path_input:
                    local_path = local_path_input
                table_name_default = f"tbl_local_{int(time.time())}"

            file_format = st.selectbox(
                "File Format:", ["parquet", "csv"], key="local_format")
            table_name = st.text_input(
                "Table Name:", table_name_default)

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
                if local_source_mode == "Upload Files" and uploaded_files:
                    # Save uploaded files to temp dir
                    with st.spinner("Saving uploaded files..."):
                        timestamp = int(time.time())
                        temp_upload_dir = os.path.join(TEMP_DATA_DIR, f"upload_{timestamp}")
                        os.makedirs(temp_upload_dir, exist_ok=True)
                        for uploaded_file in uploaded_files:
                            with open(os.path.join(temp_upload_dir, uploaded_file.name), "wb") as f:
                                f.write(uploaded_file.getbuffer())
                        local_path = temp_upload_dir
                
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
        st.markdown(f"### {ui_cfg.get('ai_section_title', '🧠 AI Data Analyst')}")

        with st.container():
            table_to_analyze = st.selectbox(
                "Select a table to analyze:", list(st.session_state['tables'].keys()))
            user_question = st.text_input(
                "Ask a question about the data:", placeholder="e.g., What's the average value of 'column_x'?")

            include_query_context = st.checkbox(
                "Include last query results in analysis", 
                value=False,
                help="If checked, the results of the last SQL query you ran (top 50 rows) will be sent to the AI as context."
            )

            if st.button("🧠 Analyze with AI"):
                if table_to_analyze and user_question:
                    with st.spinner("AI is thinking..."):
                        response_placeholder = st.empty()
                        full_response = ""
                        try:
                            # Prepare query context if available and requested
                            query_context = None
                            if include_query_context and st.session_state.get('last_query_result_df') is not None:
                                try:
                                    # Limit context size
                                    pdf = st.session_state['last_query_result_df'].limit(50).toPandas()
                                    query_context = pdf.to_csv(index=False)
                                except Exception as e: 
                                    print(f"Error preparing context: {e}")

                            stream = analyze_data_with_ai(table_to_analyze, user_question,
                                                          st.session_state['credentials'], 
                                                          st.session_state['openai_key'], 
                                                          stream=True,
                                                          query_context=query_context)
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

                        except Exception as e:
                            st.error(f"An unexpected error occurred: {e}")
                else:
                    st.warning("Please select a table and ask a question.")
        
        # Check for SQL in the latest response and offer to run it
        if st.session_state['ai_conversation']:
            latest_entry = st.session_state['ai_conversation'][0]
            sql_match = re.search(r"```sql\n(.*?)```", latest_entry['response'], re.DOTALL)
            if sql_match:
                sql_query = sql_match.group(1).strip()
                st.info("💡 SQL Query detected in the latest response.")
                if st.button("▶️ Run Extracted SQL Query", key="run_extracted_sql"):
                    st.session_state['staged_query'] = sql_query
                    st.rerun()

        # Display conversation history
        if st.session_state['ai_conversation']:
            st.markdown("---")
            st.markdown("#### History")
            for i, entry in enumerate(st.session_state['ai_conversation']):
                _display_conversation_entry(len(st.session_state['ai_conversation']) - i, entry)

if __name__ == "__main__":
    main()
