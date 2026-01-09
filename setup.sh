#!/bin/bash

# --- 1. Check for Java (Required for PySpark) ---
if ! command -v java &> /dev/null; then
    echo "❌ Error: Java is not installed. Please install Java 11 (brew install openjdk@11)."
    exit 1
fi

# --- 2. Setup Virtual Environment ---
echo "⚙️ Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# --- 3. Upgrade Pip & Setuptools ---
# This prevents many 'failed-wheel-build' errors
echo "⬆️ Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# --- 4. Install PySpark Separately ---
# We use --only-binary to avoid building from source
echo "📦 Installing PySpark (Binary)..."
pip install --only-binary=:all: pyspark==3.5.0

# --- 5. Install Remaining Requirements ---
echo "📦 Installing other dependencies..."
pip install -r requirements.txt

# --- 6. Install Local Package ---
echo "🏗️ Installing SmartSpark in editable mode..."
pip install -e .

echo "✅ Setup complete! Run the app with: source venv/bin/activate && streamlit run app.py"