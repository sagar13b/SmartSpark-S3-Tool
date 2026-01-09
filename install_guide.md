Detailed technical steps for setting up the Spark environment.

Windows Setup
Java: Download JDK 11 from Adoptium. Run the .msi.

Winutils: Spark requires winutils.exe to run on Windows.

Create a folder C:\hadoop\bin.

Download winutils.exe (Hadoop 3.3.x) and place it in that folder.

Add Environment Variable: HADOOP_HOME = C:\hadoop.

Python: pip install pyspark streamlit boto3 openai psutil

macOS Setup
Java: brew install openjdk@11

Symlink: sudo ln -sfn /usr/local/opt/openjdk@11/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-11.jdk

Python: pip install pyspark streamlit boto3 openai psutil

Linux (Ubuntu/Debian) Setup
Java: sudo apt update && sudo apt install openjdk-11-jdk -y

Python: pip install pyspark streamlit boto3 openai psutil