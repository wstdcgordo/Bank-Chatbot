import streamlit as st # type: ignore
import sqlite3
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
from datetime import datetime
import holidays
import pytz # Still used internally for holiday/date context for the AI

import logging # For backend logging

# --- Streamlit Page Configuration (MUST BE FIRST STREMLIT COMMAND) ---
st.set_page_config(page_title="Banking Assistant", page_icon="🏦", layout="centered")

# --- Logging Configuration ---
# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "app.log")

# Get the logger instance
logger = logging.getLogger(__name__) # Use __name__ to get a logger specific to this module
logger.setLevel(logging.INFO) # Set the minimum logging level (INFO, DEBUG, WARNING, ERROR, CRITICAL)

# Check if handlers are already added to prevent duplicates on Streamlit reruns
if not logger.handlers:
    # Create a file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO) # File handler will also log INFO and above

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    # Optionally, add a StreamHandler to see logs in console during development
    # (Remove or set level to WARNING for production if you don't want verbose console output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info("Application started or reloaded.") # Log when the app starts/reloads


# --- Configuration and Initialization ---
load_dotenv()
genai.configure(api_key=os.getenv("API_KEY_GM"))
model = genai.GenerativeModel("gemini-2.0-flash")

# --- Initialize Timezone (still for AI context, not displayed) ---
antipolo_tz = pytz.timezone('Asia/Manila')

# --- Removed Time Display from Sidebar ---
# The entire 'with st.sidebar:' block for the clock has been removed.
# No time will be displayed in the Streamlit UI.

# --- Initialize database connection ---
@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect("bank_transactions.db", check_same_thread=False)
    logger.info("Database connection established or retrieved from cache.")
    return conn

conn = get_db_connection()
cursor = conn.cursor()

# Get database schema
@st.cache_data
def get_schema_string(_cursor_obj):
    tables_info = []
    _cursor_obj.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in _cursor_obj.fetchall()]

    for table in tables:
        _cursor_obj.execute(f"PRAGMA table_info({table})")
        columns = _cursor_obj.fetchall()
        col_defs = ", ".join([f"{col[1]} {col[2]}" for col in columns])
        tables_info.append(f"{table}({col_defs})")
    schema_str = "\n".join(tables_info)
    logger.info("Database schema retrieved and cached.")
    return schema_str

schema = get_schema_string(cursor)

# --- Holiday Data Fetching ---
@st.cache_data
def get_philippine_holidays_cached():
    today = datetime.now()
    current_year = today.year
    next_year = today.year + 1

    ph_holidays = holidays.PH(years=[current_year, next_year])

    holiday_list = []
    for date, name in sorted(ph_holidays.items()):
        formatted_date = date.strftime('%B %d, %Y')
        holiday_list.append(f"{formatted_date}: {name}")
    logger.info("Philippine holidays fetched and cached.")
    return holiday_list

# --- Prompt Generation Function ---
def get_banking_assistant_prompt(schema_str, conversation_history_list=None, query_results_with_headers=None):
    ph_holidays = get_philippine_holidays_cached()
    holidays_str = "\n".join([f"- {h}" for h in ph_holidays])

    # Current date for context in the prompt (useful for AI)
    current_date_for_prompt = datetime.now(antipolo_tz) # Using timezone for accuracy in prompt context

    prompt_parts = [f"""
You are a friendly and intelligent banking assistant that helps users understand their financial activity by translating questions into SQL and providing clear, conversational answers — similar to how you'd reply in a chat or web interface like Gemini. You should also be ready to answer users' inquiry if they are not familiar with banking terms.

Your expertise is with bank accounts of a bank, and you're familiar with Philippine banking habits, cities (including acronyms like QC, MKT, etc.), and common transaction types (e.g., service charges, deposits, ATM withdrawals). Do not mention the bank name, but you must answer if the user asked the bank name.

Database schema:
{schema_str}

Current Date: {current_date_for_prompt.strftime('%B %d, %Y %I:%M:%S %p %Z')}

Philippine Holidays (for context, not for SQL queries unless explicitly asked about transactions on these dates):
{holidays_str}

Guidelines for SQL generation:
- When greeted, use a friendly, casual tone. Do not indicate the transactions yet.
- For each user question, arrange the date in ascending order.
- When summing amounts like Deposits or Withdrawals, always use COALESCE(column, 0) to treat NULL as zero.
- Quote column names with spaces or special characters (e.g., "Branch / Source") in SQL.
- For counts or comparisons, write WHERE conditions as needed (e.g., Balance < 30000).
- For service charges or other transaction details, match using Transaction Details like '%service charge%'.
- NULL balances should not be included in comparisons (treat as missing).
- When a user asks about transactions during a holiday, try to identify the date(s) of that holiday from the provided list.

When replying, first provide the SQL query, and then, using the results of that query, generate a friendly, conversational, and emotionally aware response. Do NOT include the SQL query in your final response to the user.

Return your response as a Python dictionary with two keys:
- "sql": The generated SQL query.
- "natural_language_response": The friendly, conversational response generated by you, based on the query results.

Important: The "natural_language_response" should be a complete, ready-to-display message. Directly incorporate the value from the SQL query results. For transaction details, always provide essential information like Date, Transaction Details, Branch, and the amount (Withdrawal or Deposit).

Example:

User: How much did I spend on service charges?

Response:
{{
  "sql": "SELECT SUM(COALESCE(Withdrawals, 0)) FROM bank_transactions WHERE LOWER(\"Transaction Details\") LIKE '%service charge%';",
  "natural_language_response": "You've spent a total of ₱500.00 on service charges. Is there anything else you'd like to check about your expenses?"
}}

Your responses should sound warm, conversational, and emotionally aware — like a smart banking assistant (e.g., Bank of America’s Erica, Axis Aha!).
Use casual phrasing where appropriate. Add context or questions to prompt further conversation (e.g., 'Want help reviewing this?', 'Let me know if that looks off!').
Avoid sounding robotic or too technical. Avoid repeating the user's question.
Be brief, helpful, and brand-friendly. For bullet forms, it must be clear, very clear.

"""]

    if conversation_history_list:
        prompt_parts.append("\n--- Conversation History ---")
        for turn in conversation_history_list:
            if turn["role"] == "user":
                prompt_parts.append(f"User: {turn['content']}")
            elif turn["role"] == "assistant":
                prompt_parts.append(f"Assistant: {turn['content']}")
        prompt_parts.append("----------------------------")

    if query_results_with_headers:
        headers = query_results_with_headers['headers']
        rows = query_results_with_headers['rows']

        formatted_results = [headers]
        for row in rows:
            formatted_row = []
            for item in row:
                if isinstance(item, (int, float)):
                    formatted_row.append(f"{item:,.2f}")
                else:
                    formatted_row.append(str(item))
            formatted_results.append(formatted_row)

        prompt_parts.append(f"\n--- SQL Query Results ---")
        prompt_parts.append(json.dumps(formatted_results, indent=2))
        prompt_parts.append("-------------------------")

    return "\n".join(prompt_parts)

# --- Gemini Interaction Function ---
def get_gemini_response(user_question, model_obj, schema_prompt_func, conversation_history_list=None, query_results_with_headers=None):
    full_prompt = f"{schema_prompt_func(schema, conversation_history_list, query_results_with_headers)}\n\nUser: {user_question}\n\nAssistant Response:"
    logger.debug(f"Sending prompt to Gemini: {full_prompt[:500]}...")

    try:
        gemini_response = model_obj.generate_content(full_prompt)
        content = gemini_response.text.strip()
        logger.debug(f"Raw Gemini response received: {content[:500]}...")

        if content.startswith("```json"):
            json_str = content.strip().split('\n', 1)[1].rsplit('```', 1)[0]
        elif content.startswith("```"):
            json_str = content.strip().split('\n', 1)[1].rsplit('```', 1)[0]
        else:
            json_str = content

        response_dict = json.loads(json_str)
        logger.info(f"Successfully parsed Gemini response into dictionary.")
        return response_dict
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from Gemini: {e}. Raw content: {content}", exc_info=True)
        st.error(f"Error decoding JSON from Gemini: {e}")
        st.code(f"Raw Gemini response:\n{content}")
        return None
    except Exception as e:
        logger.error(f"An error occurred during Gemini API call: {e}", exc_info=True)
        st.error(f"An error occurred during Gemini API call: {e}")
        return None

# --- Streamlit App Content (After set_page_config) ---
st.title("🏦 Banking Assistant")

st.write("Hi there! I'm your friendly banking assistant. How can I help you today?")
st.write("You can ask me things like 'How much did I spend last month?' or 'Show me my recent deposits.'")

if "messages" not in st.session_state:
    st.session_state.messages = []
    logger.info("Initialized new chat session (messages in session_state).")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about your transactions..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    logger.info(f"User Query: {prompt}")

    exit_phrases = ["exit", "goodbye", "thank you", "thanks", "thats all", "that's all", "bye"]
    if any(phrase in prompt.lower() for phrase in exit_phrases):
        assistant_response = "Thanks for chatting! Have a great day!"
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        logger.info(f"Chat ended with exit phrase. Assistant Response: {assistant_response}")
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_for_sql = get_gemini_response(prompt, model, get_banking_assistant_prompt,
                                                   conversation_history_list=st.session_state.messages)

            if not response_for_sql or "sql" not in response_for_sql:
                assistant_response = "I'm sorry, I couldn't generate a SQL query for that request. Could you please rephrase it?"
                st.markdown(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                logger.warning(f"Gemini failed to generate SQL for user query: {prompt}. Assistant response: {assistant_response}")
                st.stop()

            sql_query = response_for_sql["sql"]
            logger.info(f"Generated SQL Query: {sql_query}")

            try:
                cursor.execute(sql_query)
                query_results = cursor.fetchall()
                logger.info(f"SQL Query Executed Successfully. Results: {query_results}")

                column_headers = [description[0] for description in cursor.description]
                query_results_with_headers = {"headers": column_headers, "rows": query_results}

                final_response_dict = get_gemini_response(
                    prompt, model, get_banking_assistant_prompt,
                    conversation_history_list=st.session_state.messages,
                    query_results_with_headers=query_results_with_headers
                )

                if final_response_dict and "natural_language_response" in final_response_dict:
                    assistant_response_text = final_response_dict['natural_language_response']
                    st.markdown(assistant_response_text)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response_text})
                    logger.info(f"Final Assistant Response (NL): {assistant_response_text}")
                else:
                    assistant_response = "I executed the query, but I had trouble forming a clear response. Please try again."
                    st.markdown(assistant_response)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    logger.error(f"Gemini failed to generate final NL response after query execution. Assistant response: {assistant_response}")

            except sqlite3.Error as e:
                assistant_response = f"I ran into an issue processing that request. Database error: {e}. Could you try rephrasing your question?"
                st.markdown(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                logger.error(f"Database error during SQL execution for query '{sql_query}': {e}", exc_info=True)
            except Exception as e:
                assistant_response = f"An unexpected error occurred: {e}. Please try again or rephrase your question."
                st.markdown(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)