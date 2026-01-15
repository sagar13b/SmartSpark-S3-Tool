import streamlit as st
from dotenv import load_dotenv
from .utils import (
    STREAMLIT_CONFIG_FILE,
    OPENAI_API_KEY_ENV_VAR,
    AWS_CREDENTIALS_ENV_PREFIX,
    AZURE_CREDENTIALS_ENV_PREFIX,
    update_env_file,
    load_app_config
)
from .auth import (
    load_openai_key,
    validate_openai_key,
    load_credentials,
    parse_aws_credentials,
    parse_azure_credentials,
    validate_s3_credentials,
    validate_azure_credentials
)

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


def show_openai_setup():
    """Show OpenAI setup dialog, allowing user to enter key and save to .env file."""
    ui_cfg = load_app_config()
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
