import os
import sqlite3
import json
from datetime import datetime
import holidays
import pytz
import logging

from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
import google.generativeai as genai

# --- Flask App Initialization ---
app = Flask(__name__)
# IMPORTANT: Replace with a strong, random secret key for production!
app.secret_key = os.urandom(24) # Used for Flask sessions

# --- Logging Configuration ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "app.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info("Flask Application started.")

# --- Configuration and Initialization ---
load_dotenv()
genai.configure(api_key=os.getenv("API_KEY_GM"))
model = genai.GenerativeModel("gemini-2.0-flash")

# --- Initialize Timezone ---
antipolo_tz = pytz.timezone('Asia/Manila')

# --- Initialize database connection ---
def get_db_connection():
    # In Flask, we can't use @st.cache_resource directly.
    # For simplicity, we'll create a connection per request or use Flask-SQLAlchemy for ORM.
    # For this example, a simple direct connection will suffice.
    conn = sqlite3.connect("../test/bank_transactions.db")
    conn.row_factory = sqlite3.Row # This makes results accessible by column name
    logger.info("Database connection established.")
    return conn

# Get database schema
# This can be cached in a global variable or app context for efficiency
schema = None
def get_schema_string():
    global schema
    if schema is not None:
        return schema
    conn = get_db_connection()
    cursor = conn.cursor()
    tables_info = []
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]

    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        col_defs = ", ".join([f"{col[1]} {col[2]}" for col in columns])
        tables_info.append(f"{table}({col_defs})")
    schema_str = "\n".join(tables_info)
    conn.close()
    schema = schema_str # Cache it after first retrieval
    logger.info("Database schema retrieved and cached.")
    return schema_str

# Call once to cache schema on app startup
get_schema_string()


# --- Holiday Data Fetching ---
# This can also be cached globally
ph_holidays_cached = None
def get_philippine_holidays_cached():
    global ph_holidays_cached
    if ph_holidays_cached is not None:
        return ph_holidays_cached

    today = datetime.now()
    current_year = today.year
    next_year = today.year + 1

    ph_holidays = holidays.PH(years=[current_year, next_year])

    holiday_list = []
    for date, name in sorted(ph_holidays.items()):
        formatted_date = date.strftime('%B %d, %Y')
        holiday_list.append(f"{formatted_date}: {name}")
    ph_holidays_cached = holiday_list # Cache it
    logger.info("Philippine holidays fetched and cached.")
    return holiday_list

# Call once to cache holidays on app startup
get_philippine_holidays_cached()


# --- Prompt Generation Function ---
def get_banking_assistant_prompt(schema_str, preferred_language, conversation_history_list=None, query_results_with_headers=None):
    ph_holidays = get_philippine_holidays_cached()
    holidays_str = "\n".join([f"- {h}" for h in ph_holidays])
    current_date_for_prompt = datetime.now(antipolo_tz)

    language_instruction = ""
    if preferred_language == "es":
        language_instruction = "You must respond entirely in Latin American Spanish. All financial terms, greetings, and explanations should be in Latin American Spanish."
        examples_section = """
Example (English):

User: How much did I spend on service charges?

Response:
{{
  "sql": "SELECT SUM(COALESCE(Withdrawals, 0)) FROM bank_transactions WHERE LOWER(\"Transaction Details\") LIKE '%service charge%';",
  "natural_language_response": "You've spent a total of ₱500.00 on service charges. Is there anything else you'd like to check about your expenses?"
}}

Example (Spanish):

User: ¿Cuánto gasté en cargos por servicio?

Response:
{{
  "sql": "SELECT SUM(COALESCE(Withdrawals, 0)) FROM bank_transactions WHERE LOWER(\"Transaction Details\") LIKE '%service charge%';",
  "natural_language_response": "Ha gastado un total de ₱500.00 en cargos por servicio. ¿Hay algo más que le gustaría revisar sobre sus gastos?"
}}
"""
    else: # Default to English for 'en' or any other value
        language_instruction = "You must respond entirely in English. All financial terms, greetings, and explanations should be in English."
        examples_section = """
Example:

User: How much did I spend on service charges?

Response:
{{
  "sql": "SELECT SUM(COALESCE(Withdrawals, 0)) FROM bank_transactions WHERE LOWER(\"Transaction Details\") LIKE '%service charge%';",
  "natural_language_response": "You've spent a total of ₱500.00 on service charges. Is there anything else you'd like to check about your expenses?"
}}
"""


    prompt_parts = [f"""
You are a friendly and intelligent banking assistant that helps users understand their financial activity by translating questions into SQL and providing clear, conversational answers.

{language_instruction}

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
- When asked about transactions on specific ranges, try to list all of them.

When replying, first provide the SQL query, and then, using the results of that query, generate a friendly, conversational, and emotionally aware response. Do NOT include the SQL query in your final response to the user.

Return your response as a Python dictionary with two keys:
- "sql": The generated SQL query.
- "natural_language_response": The friendly, conversational response generated by you, based on the query results.

{examples_section}

Your responses should sound warm, conversational, and emotionally aware — like a smart banking assistant (e.g., Bank of America’s Erica, Axis Aha!).
Use casual phrasing where appropriate. Add context or questions to prompt further conversation (e.g., 'Want help reviewing this!', 'Let me know if that looks off!').
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
def get_gemini_response(user_question, model_obj, preferred_language, conversation_history_list=None, query_results_with_headers=None):
    # Use the globally cached schema
    full_prompt = f"{get_banking_assistant_prompt(get_schema_string(), preferred_language, conversation_history_list, query_results_with_headers)}\n\nUser: {user_question}\n\nAssistant Response:"
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
        return {"error": f"Error decoding JSON from Gemini: {e}", "raw_content": content}
    except Exception as e:
        logger.error(f"An error occurred during Gemini API call: {e}", exc_info=True)
        return {"error": f"An error occurred during Gemini API call: {e}"}

# --- Flask Routes ---

@app.route('/')
def index():
    # Initialize conversation history in session if not present
    if 'messages' not in session:
        session['messages'] = []
        logger.info("Initialized new chat session (messages in Flask session).")
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json.get('query')
    preferred_language = request.json.get('language', 'en') # Get language from frontend, default to 'en'
    if not user_query:
        logger.warning("Received empty query from frontend.")
        return jsonify({"response": "Please enter a query."}), 400

    logger.info(f"User Query ({preferred_language}): {user_query}")

    # Retrieve messages from session (Flask's equivalent of st.session_state)
    messages = session.get('messages', [])

    # Append user query to history
    messages.append({"role": "user", "content": user_query})

    # Check for exit phrases (can be localized too)
    exit_phrases_en = ["exit", "goodbye", "thank you", "thanks", "thats all", "that's all", "bye"]
    exit_phrases_es = ["salir", "adiós", "gracias", "eso es todo", "hasta luego"]
    
    if (preferred_language == "en" and any(phrase in user_query.lower() for phrase in exit_phrases_en)) or \
       (preferred_language == "es" and any(phrase in user_query.lower() for phrase in exit_phrases_es)):
        if preferred_language == "es":
            assistant_response_text = "¡Gracias por chatear! ¡Que tengas un gran día!"
        else:
            assistant_response_text = "Thanks for chatting! Have a great day!"
        
        messages.append({"role": "assistant", "content": assistant_response_text, "sql": None})
        session['messages'] = messages
        logger.info(f"Chat ended with exit phrase. Assistant Response: {assistant_response_text}")
        return jsonify({"response": assistant_response_text, "sql_query": None})

    sql_query = None

    response_for_sql = get_gemini_response(user_query, model, preferred_language, conversation_history_list=messages)

    if response_for_sql and "sql" in response_for_sql:
        sql_query = response_for_sql["sql"]
        logger.info(f"Generated SQL Query: {sql_query}")
    else:
        if preferred_language == "es":
            assistant_response_text = "Lo siento, no pude generar una consulta SQL para esa solicitud. ¿Podrías reformularla?"
        else:
            assistant_response_text = "I'm sorry, I couldn't generate a SQL query for that request. Could you please rephrase it?"
        
        messages.append({"role": "assistant", "content": assistant_response_text, "sql": None})
        session['messages'] = messages
        logger.warning(f"Gemini failed to generate SQL for user query: {user_query}. Assistant response: {assistant_response_text}")
        return jsonify({"response": assistant_response_text, "sql_query": None})

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql_query)
        query_results = cursor.fetchall()
        conn.close()
        logger.info(f"SQL Query Executed Successfully. Results: {query_results}")

        column_headers = [description[0] for description in cursor.description]
        query_results_with_headers = {"headers": column_headers, "rows": [list(row) for row in query_results]}

        final_response_dict = get_gemini_response(
            user_query, model,
            preferred_language, # Pass preferred language to the second Gemini call
            conversation_history_list=messages,
            query_results_with_headers=query_results_with_headers
        )

        if final_response_dict and "natural_language_response" in final_response_dict:
            assistant_response_text = final_response_dict['natural_language_response']
            logger.info(f"Final Assistant Response (NL): {assistant_response_text}")
        else:
            if preferred_language == "es":
                assistant_response_text = "Ejecuté la consulta, pero tuve problemas para formar una respuesta clara. Por favor, inténtalo de nuevo."
            else:
                assistant_response_text = "I executed the query, but I had trouble forming a clear response. Please try again."
            logger.error(f"Gemini failed to generate final NL response after query execution. Assistant response: {assistant_response_text}")
            sql_query = None

    except sqlite3.Error as e:
        if preferred_language == "es":
            assistant_response_text = f"Tuve un problema al procesar esa solicitud. Error de base de datos: {e}. ¿Podrías reformular tu pregunta?"
        else:
            assistant_response_text = f"I ran into an issue processing that request. Database error: {e}. Could you try rephrasing your question?"
        logger.error(f"Database error during SQL execution for query '{sql_query}': {e}", exc_info=True)
        sql_query = None
    except Exception as e:
        if preferred_language == "es":
            assistant_response_text = f"Ocurrió un error inesperado: {e}. Por favor, inténtalo de nuevo o reformula tu pregunta."
        else:
            assistant_response_text = f"An unexpected error occurred: {e}. Please try again or rephrase your question."
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
        sql_query = None

    # Append assistant response to history
    messages.append({"role": "assistant", "content": assistant_response_text, "sql": sql_query})
    session['messages'] = messages

    return jsonify({"response": assistant_response_text, "sql_query": sql_query})

if __name__ == '__main__':
    app.run(debug=True, port=5001)