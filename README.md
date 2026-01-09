# 🔍 SmartSpark S3 AI Explorer

**SmartSpark** is a high-performance data workbench that bridges the gap between **AWS S3 Data Lakes** and **OpenAI Analysis**. By leveraging **Apache Spark**, it allows you to query massive Parquet and CSV files locally using SQL, while an integrated AI agent helps interpret results and write queries for you.

---

## ✨ Key Features

* **S3 Native:** Seamlessly pull and cache data from AWS S3 buckets.
* **Spark Engine:** Perform distributed SQL queries locally with industry-standard performance.
* **AI Data Analyst:** Powered by GPT-4o-mini to write SQL, identify anomalies, and summarize findings.
* **Fully Customizable:** Change branding and AI behavior via JSON—**no coding required.**
* **Resource Monitor:** Real-time tracking of CPU, RAM, and Disk usage.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure you have the following installed on your system:
* **Java 11:** Required to run the Apache Spark engine.
* **Python 3.9+:** The primary programming environment.
* **Hadoop Binaries (Windows Only):** Download `winutils.exe` for Hadoop 3.3, place it in `C:\hadoop\bin`, and set the `HADOOP_HOME` environment variable to `C:\hadoop`.

### 2. Environment Setup & Installation
We use a **Virtual Environment (venv)** to keep your global Python installation clean and a **setup.py** file to automate dependency installation.

```bash
# Clone the repository
git clone https://github.com/sagar13b/SmartSpark-S3-Tool.git
cd SmartSpark-S3-Tool

# Create a virtual environment (Isolated Workspace)
python -m venv venv

# Activate the environment
# On Mac/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# Install the app using setup.py
# The '-e' flag allows you to edit the code and see changes instantly
pip install -e .

### 3. Launch the App
Open your terminal, ensure your virtual environment is activated, and run:
```bash
streamlit run app.py

This will start the SmartSpark web application. A new browser window should automatically open displaying the interface. If it doesn't, navigate to the URL shown in your terminal (typically http://localhost:8501).

## 🛠️ Internal Setup & Architecture

### Why Virtual Environments?
A virtual environment ensures that the specific versions of pyspark, streamlit, and openai used in this project do not conflict with other Python projects on your computer. It creates a local "sandbox" just for SmartSpark.

### Why setup.py?
Unlike a simple `requirements.txt` file, `setup.py` treats SmartSpark as a professional Python package. It manages:

* **Dependency Resolution:** Ensuring all libraries work together.
* **Version Control:** Automatically fetching the correct library versions.
* **Local Linking:** Allowing the app to recognize internal modules easily.

---

## ⚙️ Configuration Templates

Create these files in your root directory to customize the application.

### 🎨 App UI (config.json)
```json
{
  "app_title": "SmartSpark Explorer",
  "header_title": "🚀 SmartSpark Data Workbench",
  "sidebar_header": "Data Controls",
  "theme_color": "#FF4B4B"
}

### 🧠 AI Logic (prompts.json)
```json
{
  "system_message": "You are a Senior Data Analyst. Help users write Spark SQL and find insights.",
  "one_shot_example": "User: Top 5 sales. Assistant: SELECT * FROM df ORDER BY sales DESC LIMIT 5;",
  "analysis_instruction": "Identify trends and anomalies in the provided data sample."
}

### 🔑 Credentials (aws_credentials.json)
```json
{
  "aws_access_key": "YOUR_ACCESS_KEY",
  "aws_secret_key": "YOUR_SECRET_KEY",
  "region": "us-east-1"
}

## 🔒 Security & Privacy

Your data security is a priority. The app is configured with a `.gitignore` to ensure your sensitive keys never leave your machine:

* `aws_credentials.json`: Stores S3 Access keys.
* `openai_key.json`: Stores your OpenAI API key.
* `spark_tables_metadata.json`: Local cache tracking.

## 🛠️ Troubleshooting

* **Spark Errors:** Ensure `JAVA_HOME` is pointing to Java 11.
* **Windows FileSystem:** Ensure `winutils.exe` is present in your Hadoop path.
* **S3 Access:** Check that your IAM user has `s3:ListBucket` and `s3:GetObject` permissions.
* **App Launch Issues:** Make sure your virtual environment is activated and all dependencies are installed correctly.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any features or bug fixes.