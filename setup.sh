#!/bin/bash
echo "🚀 Initializing SmartSpark Repository..."

# Create Virtual Env
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install --upgrade pip
pip install streamlit pandas pyspark boto3 botocore psutil openai

# 3. Check for Java
if type -p java > /dev/null; then
    echo "Java found: $(java -version 2>&1 | head -n 1)"
else
    echo "WARNING: Java not found. Please install JDK 11 to use Spark."
fi

# Create placeholder config if missing
if [ ! -f config.json ]; then
  echo '{"app_title": "My Spark Tool"}' > config.json
fi

echo "✅ Ready! Run: source venv/bin/activate && streamlit run app.py"