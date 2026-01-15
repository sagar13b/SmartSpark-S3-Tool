import os
import json
import streamlit as st
from openai import OpenAI
from .utils import PROMPTS_FILE
from .spark_manager import get_spark_session

def get_table_summary_spark(table_name, credentials):
    """Get comprehensive summary of table for AI context using pure Spark"""
    try:
        table_info = st.session_state['tables'][table_name]
        # spark = get_spark_session(credentials) # Not strictly needed if we use df directly
        df = table_info['dataframe']

        # Get sample data using Spark
        sample_rows = df.limit(5).collect()
        sample_data = "\n".join([str(row.asDict()) for row in sample_rows])

        # Get numeric columns
        numeric_cols = [f.name for f in df.schema.fields if str(f.dataType) in [
            'IntegerType', 'LongType', 'FloatType', 'DoubleType', 'DecimalType']]

        stats_summary = ""
        if numeric_cols:
            stats_df = df.select(numeric_cols).summary(
                "count", "mean", "stddev", "min", "max")
            stats_rows = stats_df.collect()
            stats_summary = "\n".join([str(row.asDict())
                                      for row in stats_rows])

        summary = f"""
TABLE: {table_name}
Total Rows: {table_info['row_count']:,}
Total Columns: {table_info['col_count']}

SCHEMA:
{chr(10).join([f"- {col['name']}: {col['type']}" for col in table_info['schema']])}

SAMPLE DATA (first 5 rows):
{sample_data}

NUMERIC COLUMN STATISTICS:
{stats_summary if stats_summary else "No numeric columns"}
"""
        return summary
    except Exception as e:
        return f"Error getting table summary: {str(e)}"


def execute_query_for_ai(query, credentials):
    """Execute a SQL query and return results as string using Spark"""
    try:
        spark = get_spark_session(credentials)
        result = spark.sql(query)
        rows = result.limit(100).collect()
        result_str = "\n".join([str(row.asDict()) for row in rows])
        return True, result_str, len(rows)
    except Exception as e:
        return False, str(e), 0


def load_prompts():
    """Load AI prompts from external JSON file with fallback defaults"""
    default_prompts = {
        "system_message": "You are a professional Data Analyst with expertise in SQL and data analysis.",
        "one_shot_example": "Example: When asked 'What are the top 5 products by revenue?', analyze the schema and provide: 'Based on the data, I'll write a query to find the top 5 products:\n```sql\nSELECT product_name, SUM(revenue) as total_revenue\nFROM sales_table\nGROUP BY product_name\nORDER BY total_revenue DESC\nLIMIT 5\n```\nThis query groups sales by product and returns the highest earners.'",
        "analysis_instruction": "Analyze this table: {table_name}\n\nUser Question: {user_question}\n\nProvide a clear answer. If you need to query the data, include a SQL query in a ```sql code block.",
        "results_interpretation": "Query Results:\n{query_result}\n\nOriginal Question: {user_question}\n\nInterpret these results and provide actionable insights."
    }
    try:
        if os.path.exists(PROMPTS_FILE):
            with open(PROMPTS_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading prompts.json: {e}")
    return default_prompts


def analyze_data_with_ai(table_name, user_question, credentials, api_key, stream=False, query_context=None):
    """
    Analyzes data using OpenAI's GPT-4o-mini model.
    Uses pure Spark instead of pandas
    """
    try:
        client = OpenAI(api_key=api_key)
        prompts = load_prompts()

        # Get richer table context
        # Check if table exists in st.session_state
        if 'tables' not in st.session_state or table_name not in st.session_state['tables']:
             raise ValueError(f"Table {table_name} not found in session state")
             
        df = st.session_state['tables'][table_name]['dataframe']
        table_info = st.session_state['tables'][table_name]

        # Build comprehensive context using Spark
        schema_str = "\n".join(
            [f"  - {col['name']}: {col['type']}" for col in table_info['schema']])

        # Get sample data using Spark collect
        sample_rows = df.limit(10).collect()
        sample_str = "\n".join([str(row.asDict()) for row in sample_rows])

        context = f"""Table: {table_name}
Total Rows: {table_info['row_count']:,}
Total Columns: {table_info['col_count']}

Schema:
{schema_str}

Sample Data (first 10 rows):
{sample_str}
"""
        
        additional_context = ""
        if query_context:
            additional_context = f"\nRELEVANT QUERY CONTEXT (Use this if helpful):\n{query_context}\n"

        # Build messages
        system_msg = f"{prompts['system_message']}\n\nAVAILABLE DATA:\n{context}"
        if prompts.get("one_shot_example"):
            system_msg += f"\n\n{prompts['one_shot_example']}"

        user_msg = prompts["analysis_instruction"].format(
            table_name=table_name,
            user_question=user_question,
            additional_context=additional_context
        )

        if stream:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.7,
                stream=True
            )
        else:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
    except Exception as e:
        if stream:
            # We cannot yield from here easily if it's not a generator function normally. 
            # But the caller expects a generator or iterable.
            # We wrap it in a list to properly return an iterable with an error dict
            return [{"error": str(e)}]
        else:
            return f"Error: {str(e)}"


def analyze_with_query_results(table_name, user_question, query_result, credentials, api_key):
    """
    Performs follow-up analysis on SQL query results using OpenAI's GPT-4o-mini model.
    """
    try:
        client = OpenAI(api_key=api_key)
        prompts = load_prompts()

        system_prompt = prompts["results_interpretation"].format(
            query_result=query_result,
            user_question=user_question
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"
