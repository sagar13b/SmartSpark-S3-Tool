# 🔍 SmartSpark S3 AI Explorer

A Streamlit-powered workbench for querying S3 data lakes using Apache Spark and OpenAI.

## 🚀 Setup
1. **Prerequisites:** Ensure Java 11 and Python 3.9+ are installed (see `install_guide.md`).
2. **Install Deps:** `pip install -r requirements.txt`
3. **Launch:** `streamlit run app.py`

## 📂 Features
- **S3 Connectivity:** Seamlessly pull Parquet/CSV from AWS.
- **Local Caching:** Automatic local metadata and data storage for speed.
- **AI Analyst:** Chat with your data. The AI writes SQL and interprets results for you.
- **Resource Monitor:** View CPU and RAM usage in real-time.

> **Note:** Your credentials are saved locally in `aws_credentials.json` and `openai_key.json`. These are ignored by git for security.