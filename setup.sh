#!/bin/bash

# 1. Create Virtual Environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Install Python Dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install streamlit pandas pyspark boto3 botocore psutil openai

# 3. Check for Java
if type -p java > /dev/null; then
    echo "Java found: $(java -version 2>&1 | head -n 1)"
else
    echo "WARNING: Java not found. Please install JDK 11 to use Spark."
fi

echo "Setup complete. Start the app with: streamlit run app.py"