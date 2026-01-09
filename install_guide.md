# 🚀 SmartSpark Installation Guide

## Prerequisites

### 1. Required Software
Ensure you have the following installed on your system:

* **Java 11:** Required to run the Apache Spark engine
  - Download from [Oracle Java 11](https://www.oracle.com/java/technologies/javase/jdk11-archive-downloads.html) or [OpenJDK 11](https://adoptium.net/temurin/releases/?version=11)
  - Set `JAVA_HOME` environment variable

* **Python 3.9+:** The primary programming environment
  - Download from [python.org](https://www.python.org/downloads/)
  - Ensure Python is added to your system PATH

* **Git:** For cloning the repository
  - Download from [git-scm.com](https://git-scm.com/downloads)

### 2. Windows-Specific Requirements (Only for Windows Users)
If you're using Windows, you need Hadoop binaries:

1. Download `winutils.exe` for Hadoop 3.3
2. Create directory: `C:\hadoop\bin`
3. Place `winutils.exe` in `C:\hadoop\bin`
4. Set environment variable: `HADOOP_HOME=C:\hadoop`

---

## 📦 Installation Steps

### Step 1: Clone the Repository
```bash
git clone https://github.com/sagar13b/SmartSpark-S3-Tool.git
cd SmartSpark-S3-Tool
```

### Step 2: Create Virtual Environment
Create an isolated Python environment to avoid dependency conflicts:

```
For macOS/Linux:
python3 -m venv venv
source venv/bin/activate

For Windows:
CODE_BLOCK_START
python -m venv venv
.\venv\Scripts\activate
```


### Step 3: Install Dependencies
Install SmartSpark and all required packages:

Install in development mode (allows code editing):

```
pip install -e .
```

Alternative: Install using requirements.txt (if available):
```
pip install -r requirements.txt
```

### Step 4: Verify Installation
Check if all packages are installed correctly:

```
pip list | grep -E "(pyspark|streamlit|openai)"
```

You should see:
- pyspark>=3.5.0
- streamlit>=1.28.0
- openai>=1.3.0