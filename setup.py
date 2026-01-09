from setuptools import setup, find_packages

setup(
    name="smartspark",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyspark==3.5.0",
        "streamlit",
        "openai",
        "pandas",
        "boto3",
        "python-dotenv",
        "psutil"
    ],
    entry_points={
        "console_scripts": [
            "smartspark=app:main",
        ],
    },
)