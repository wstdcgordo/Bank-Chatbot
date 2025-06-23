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

# --- Global Caches (for schema and holidays) ---
_cached_schema = None
_cached_ph_holidays = None

# --- Database Connection ---
def get_db_connection():
    """Establishes and returns a SQLite database connection."""
    # Ensure this points to your actual DB file.
    conn = sqlite3.connect("bank_transactions(1).db")
    conn.row_factory = sqlite3.Row # This makes results accessible by column name
    logger.debug("Database connection established.")
    return conn

# --- Database Schema Retrieval ---
def get_schema_string():
    """Retrieves and caches the database schema string."""
    global _cached_schema
    if _cached_schema is not None:
        logger.debug("Returning schema from cache.")
        return _cached_schema

    conn = get_db_connection()
    cursor = conn.cursor()
    tables_info = []
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]

    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        col_defs = ", ".join([f'"{col[1]}" {col[2]}' for col in columns])
        tables_info.append(f"{table}({col_defs})")
    schema_str = "\n".join(tables_info)
    conn.close()
    _cached_schema = schema_str # Cache it after first retrieval
    logger.info("Database schema retrieved and cached (newly).")
    return schema_str

# --- Holiday Data Fetching ---
def get_philippine_holidays_cached():
    """Fetches and caches Philippine holidays."""
    global _cached_ph_holidays
    if _cached_ph_holidays is not None:
        logger.debug("Returning holidays from cache.")
        return _cached_ph_holidays

    today = datetime.now()
    current_year = today.year
    next_year = today.year + 1

    ph_holidays = holidays.PH(years=[current_year, next_year])

    holiday_list = []
    for date, name in sorted(ph_holidays.items()):
        formatted_date = date.strftime('%B %d, %Y')
        holiday_list.append(f"{formatted_date}: {name}")
    _cached_ph_holidays = holiday_list # Cache it
    logger.info("Philippine holidays fetched and cached (newly).")
    return holiday_list

# Call to ensure initial cache population
get_schema_string()
get_philippine_holidays_cached()

## 1. SQL Generation Prompt (`get_sql_generation_prompt`)

def get_sql_generation_prompt(schema_str, preferred_language, conversation_history_list=None):
    """
    Generates the prompt specifically for the LLM to create SQL queries.
    This prompt strictly focuses on database schema and SQL generation rules.
    """
    ph_holidays = get_philippine_holidays_cached()
    holidays_str = "\n".join([f"- {h}" for h in ph_holidays])
    current_date_for_prompt = datetime.now(antipolo_tz)

    language_instruction = ""
    if preferred_language == "es":
        language_instruction = "You must generate the SQL query according to Latin American Spanish context if applicable, but SQL syntax must remain English."
        examples_section = """
Example (English):

User: How much did I spend on service charges?

Response:
{{
  "sql": "SELECT SUM(COALESCE(Withdrawals, 0)) FROM bank_transactions WHERE LOWER(\"Transaction Details\") LIKE '%service charge%';",
  "natural_language_response": "SQL generated, awaiting execution."
}}

Example (Spanish):

User: ¿Cuánto gasté en cargos por servicio?

Response:
{{
  "sql": "SELECT SUM(COALESCE(Withdrawals, 0)) FROM bank_transactions WHERE LOWER(\"Transaction Details\") LIKE '%service charge%';",
  "natural_language_response": "SQL generado, esperando ejecución."
}}
"""
    else: # Default to English for 'en' or any other value
        language_instruction = "You must generate the SQL query according to standard English."
        examples_section = """
Example:

User: How much did I spend on service charges?

Response:
{{
  "sql": "SELECT SUM(COALESCE(Withdrawals, 0)) FROM bank_transactions WHERE LOWER(\"Transaction Details\") LIKE '%service charge%';",
  "natural_language_response": "SQL generated, awaiting execution."
}}
"""

    # This section remains for SQL generation to understand common terms for filtering
    sql_query_filtering_guidelines = """
SQL Query Filtering Guidelines:
When the user asks about general categories, infer the following common Philippine-specific terms for the "Transaction Details" column. Use `LOWER(\"Transaction Details\") LIKE '%keyword%' OR LOWER(\"Transaction Details\") LIKE '%another keyword%'` to capture these:

-   **Transportation/Transpo/Rides:** 'angkas', 'moveit', 'grab', 'taxi', 'jeepney', 'bus', 'mrt', 'lrt', 'fare', 'transport'
-   **Food/Dining/Restaurants:** 'jollibee', 'mcdo', 'mcdonalds', 'mang inasal', 'chowking', 'goldilocks', 'red ribbon', 'foodpanda', 'grabfood', 'restaurant', 'cafe', 'dining', 'food', 'meal'
-   **Utilities/Bills:** 'meralco', 'maynilad', 'manila water', 'converge', 'pldt', 'globe', 'sky cable', 'internet', 'electricity', 'water bill', 'utility'
-   **Shopping/Groceries:** 'sm store', 'sm supermarket', 'ayala malls', 'robinsons', 'puregold', 'waltermart', 'rustans', 'lazada', 'shopee', 'shopping', 'grocery', 'supermarket', 'mall'
-   **Remittances/Transfers:** 'palawan express', 'lbc', 'gcash', 'paymaya', 'instapay', 'pesonet', 'money transfer', 'remittance', 'send money'
-   **Cash Withdrawals:** 'atm', 'cash withdrawal', 'cashout'
-   **Loan Payments:** 'loan payment', 'loan amortization', 'interest payment'
-   **Service Charges/Bank Fees:** 'service charge', 'bank fee', 'monthly fee', 'atm fee'

Always use `LOWER("Transaction Details")` for case-insensitive matching.
"""

    prompt_parts = [f"""
You are an intelligent assistant that specializes in translating natural language questions into SQL queries for a bank transaction database. Your sole task is to generate a correct and efficient SQL query.

{language_instruction}

Your expertise is with bank accounts of a bank. Do not mention the bank name, but you must answer if the user asked the bank name.

Database schema:
{schema_str}

Current Date: {current_date_for_prompt.strftime('%B %d, %Y %I:%M:%S %p %Z')}

Philippine Holidays (for date-related queries):
{holidays_str}

{sql_query_filtering_guidelines}

Guidelines for SQL generation:
- For each user question, arrange the date in ascending order.
- When summing amounts like Deposits or Withdrawals, always use COALESCE(column, 0) to treat NULL as zero.
- Quote column names with spaces or special characters (e.g., "Branch / Source") in SQL.
- For counts or comparisons, write WHERE conditions as needed (e.g., Balance < 30000).
- For service charges or other transaction details, match using Transaction Details like '%service charge%'.
- NULL balances should not be included in comparisons (treat as missing).
- When a user asks about transactions during a holiday, identify the date(s) of that holiday from the provided list to use in the SQL query.
- When asked about transactions on specific ranges, try to list all of them.
- Ensure the SQL query is valid for SQLite.

Return your response as a Python dictionary with two keys:
- "sql": The generated SQL query.
- "natural_language_response": "SQL generated, awaiting execution." (This is a placeholder for the first stage).
{examples_section}
"""]
    if conversation_history_list:
        prompt_parts.append("\n--- Conversation History ---")
        for turn in conversation_history_list:
            if turn["role"] == "user":
                prompt_parts.append(f"User: {turn['content']}")
            elif turn["role"] == "assistant":
                # For SQL generation, assistant content is typically the previous SQL query/response format
                # We extract the SQL part if available to ensure model learns from past SQL
                if "sql" in turn and turn["sql"]:
                     prompt_parts.append(f"Assistant: {json.dumps({'sql': turn['sql'], 'natural_language_response': 'SQL generated, awaiting execution.'}, indent=2)}")
                else:
                    prompt_parts.append(f"Assistant: {json.dumps({'sql': 'N/A', 'natural_language_response': turn['content']}, indent=2)}")
        prompt_parts.append("----------------------------")

    return "\n".join(prompt_parts)

## 2. Natural Language Response Prompt (`get_natural_language_response_prompt`)

def get_natural_language_response_prompt(user_question, schema_str, preferred_language, conversation_history_list, query_results_with_headers):
    """
    Generates the prompt for the LLM to interpret SQL results and
    create a friendly, contextual natural language response.
    """
    current_date_for_prompt = datetime.now(antipolo_tz)

    language_instruction = ""
    if preferred_language == "es":
        language_instruction = "You must respond entirely in Latin American Spanish. All financial terms, greetings, and explanations should be in Latin American Spanish."
    else:
        language_instruction = "You must respond entirely in English. All financial terms, greetings, and explanations should be in English."

    # This section contains the contextualization logic for NL response
    nl_contextualization_guidelines = """
Natural Language Response Contextualization:
Your primary task is to interpret the provided SQL query results in a friendly, conversational, and emotionally aware manner. Use your vast general knowledge to add context, especially for Philippine-specific entities, even if those specific details (like exact merchant category/location) are not explicit columns in the database.

- **Leverage General Knowledge for Context:** If specific columns for merchant type (e.g., 'Restaurant', 'Retail Store') or exact location (e.g., 'Glorietta branch') are NOT present in the database schema or query results, **use your general knowledge about common Philippine merchants and services to infer and provide relevant context based on "Transaction Details" or "Branch / Source" from the query results.**
- **Inferring Merchant Types:**
    - If "Transaction Details" contains "Jollibee", you can infer it's a fast-food chain.
    - If "Transaction Details" mentions "Grab Ride" or "Angkas", infer it's a ride-sharing service.
    - If "Transaction Details" includes "Meralco", infer it's an electricity utility.
    - If a merchant is commonly known as a grocery store (e.g., "Puregold", "SM Supermarket"), you can contextualize it as such.
- **Inferring Locations/Sources:**
    - If "Branch / Source" is "SM Megamall" or "Glorietta", you can infer it's a large shopping mall/commercial center.
    - Understand that "Branch / Source" might also indicate a payment channel (e.g., "GCash", "Online Transfer") rather than a physical store location, and explain that if appropriate.
- **Manage Expectations:** If you can't provide specific details due to lack of data (even with general knowledge), politely state that the information isn't available in the transaction details.

General NL Response Guidelines:
- Be warm, conversational, and emotionally aware.
- Use casual phrasing where appropriate.
- Add context or questions to prompt further conversation (e.g., 'Want help reviewing this!', 'Let me know if that looks off!').
- Avoid sounding robotic or too technical.
- Be brief, helpful, and brand-friendly.
- For bullet forms, it must be clear, very clear.
"""
    prompt_parts = [f"""
You are a friendly and intelligent banking assistant. Your goal is to provide clear, conversational, and insightful answers based on the provided SQL query results.

{language_instruction}

Database schema (for understanding data structure, not for SQL generation in this stage):
{schema_str}

Current Date: {current_date_for_prompt.strftime('%B %d, %Y %I:%M:%S %p %Z')}

--- User's Original Question ---
{user_question}

{nl_contextualization_guidelines}

Return your response as a Python dictionary with one key:
- "natural_language_response": The friendly, conversational response generated by you, based on the query results.
"""]

    # Include conversation history (for overall context)
    if conversation_history_list:
        prompt_parts.append("\n--- Conversation History ---")
        for turn in conversation_history_list:
            if turn["role"] == "user":
                prompt_parts.append(f"User: {turn['content']}")
            elif turn["role"] == "assistant":
                # For NL generation, just the previous natural language response is relevant
                prompt_parts.append(f"Assistant: {turn['content']}")
        prompt_parts.append("----------------------------")

    # Always include SQL Query Results for this prompt
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

        prompt_parts.append(f"\n--- SQL Query Results (from database) ---")
        prompt_parts.append(json.dumps(formatted_results, indent=2))
        prompt_parts.append("-----------------------------------------")
    else:
        # This case should ideally not happen for NL generation, but good to handle
        prompt_parts.append("\n--- No SQL Query Results Provided ---")

    return "\n".join(prompt_parts)

## 3. Gemini Interaction Helper (`get_gemini_json_response`)
# # --- Gemini Interaction Function ---
def get_gemini_json_response(full_prompt, model_obj):
    """Helper to send prompt to Gemini and parse JSON response."""
    logger.debug(f"Sending prompt to Gemini: {full_prompt[:800]}...")
    try:
        gemini_response = model_obj.generate_content(full_prompt)
        content = gemini_response.text.strip()
        logger.debug(f"Raw Gemini response received: {content[:800]}...")

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
    preferred_language = request.json.get('language', 'en')
    if not user_query:
        logger.warning("Received empty query from frontend.")
        return jsonify({"response": "Please enter a query."}), 400

    logger.info(f"User Query ({preferred_language}): {user_query}")

    # Retrieve messages from session (Flask's equivalent of st.session_state)
    messages = session.get('messages', [])

    # Append user query to history immediately for context in both stages
    messages.append({"role": "user", "content": user_query})
    session['messages'] = messages # Update session

    # Check for exit phrases (can be localized too)
    exit_phrases_en = ["exit", "goodbye", "thank you", "thanks", "thats all", "that's all", "bye"]
    exit_phrases_es = ["salir", "adiós", "gracias", "eso es todo", "hasta luego"]
    
    if (preferred_language == "en" and any(phrase in user_query.lower() for phrase in exit_phrases_en)) or \
       (preferred_language == "es" and any(phrase in user_query.lower() for phrase in exit_phrases_es)):
        if preferred_language == "es":
            assistant_response_text = "¡Gracias por chatear! ¡Que tengas un gran día!"
        else:
            assistant_response_text = "Thanks for chatting! Have a great day!"
        
        # Clear session for a new conversation after exit
        session.pop('messages', None)
        logger.info(f"Chat ended with exit phrase. Assistant Response: {assistant_response_text}. Session cleared.")
        return jsonify({"response": assistant_response_text, "sql_query": None})

    sql_query = None
    assistant_response_text = ""

    # --- Stage 1: Generate SQL Query ---
    current_schema = get_schema_string()
    sql_prompt_str = get_sql_generation_prompt(current_schema, preferred_language, conversation_history_list=messages)
    
    response_for_sql = get_gemini_json_response(sql_prompt_str, model)

    if response_for_sql and "sql" in response_for_sql:
        sql_query = response_for_sql["sql"]
        logger.info(f"Generated SQL Query: {sql_query}")
    elif "error" in response_for_sql:
        if preferred_language == "es":
            assistant_response_text = "Lo siento, tuve un problema al entender tu solicitud o al generar la consulta SQL. Por favor, intenta reformularla."
        else:
            assistant_response_text = "I'm sorry, I had trouble understanding your request or generating the SQL query. Please try rephrasing it."
        logger.error(f"Gemini failed at SQL generation step for user query: {user_query}. Error: {response_for_sql.get('error')}. Raw: {response_for_sql.get('raw_content')}")
        # Append assistant response to history before returning
        messages.append({"role": "assistant", "content": assistant_response_text, "sql": None})
        session['messages'] = messages
        return jsonify({"response": assistant_response_text, "sql_query": None})
    else:
        if preferred_language == "es":
            assistant_response_text = "Lo siento, no pude generar una consulta SQL para esa solicitud. ¿Podrías reformularla?"
        else:
            assistant_response_text = "I'm sorry, I couldn't generate a SQL query for that request. Could you please rephrase it?"
        logger.warning(f"Gemini returned an unexpected structure at SQL generation for user query: {user_query}. Response: {response_for_sql}")
        # Append assistant response to history before returning
        messages.append({"role": "assistant", "content": assistant_response_text, "sql": None})
        session['messages'] = messages
        return jsonify({"response": assistant_response_text, "sql_query": None})

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql_query)
        query_results = cursor.fetchall()
        conn.close()
        logger.info(f"SQL Query Executed Successfully. Results count: {len(query_results)}")

        column_headers = [description[0] for description in cursor.description]
        query_results_with_headers = {"headers": column_headers, "rows": [list(row) for row in query_results]}

        # --- Stage 2: Generate Natural Language Response ---
        nl_prompt_str = get_natural_language_response_prompt(
            user_question=user_query, # Ensure this matches the parameter name in get_natural_language_response_prompt
            schema_str=current_schema,
            preferred_language=preferred_language,
            conversation_history_list=messages, # Pass full history including user's current query
            query_results_with_headers=query_results_with_headers # This is crucial for NL response
        )
        final_response_dict = get_gemini_json_response(nl_prompt_str, model)

        if final_response_dict and "natural_language_response" in final_response_dict:
            assistant_response_text = final_response_dict['natural_language_response']
            logger.info(f"Final Assistant Response (NL): {assistant_response_text}")
        elif "error" in final_response_dict:
            if preferred_language == "es":
                assistant_response_text = "Ejecuté la consulta, pero tuve problemas para formar una respuesta clara. Por favor, intenta de nuevo o reformula tu pregunta."
            else:
                assistant_response_text = "I executed the query, but I had trouble forming a clear response. Please try again or rephrase your question."
            logger.error(f"Gemini failed at NL generation step after query execution. Error: {final_response_dict.get('error')}. Raw: {final_response_dict.get('raw_content')}")
        else:
            if preferred_language == "es":
                assistant_response_text = "Ejecuté la consulta, pero tuve problemas para formar una respuesta clara. Por favor, intenta de nuevo o reformula tu pregunta."
            else:
                assistant_response_text = "I executed the query, but I had trouble forming a clear response. Please try again or rephrase your question."
            logger.error(f"Gemini returned an unexpected structure at NL generation after query execution. Response: {final_response_dict}")

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