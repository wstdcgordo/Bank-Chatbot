import os
import sqlite3
import json
from datetime import datetime
import holidays
import pytz
import logging
from collections import OrderedDict 

from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
import google.generativeai as genai

# --- Import the evaluation module ---
import evaluation_module 

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Logging Configuration ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, "app.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info("Flask Application started.")

# --- Evaluation Output Directory Configuration ---
EVALUATION_OUTPUT_DIR = "evaluation_results"
os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)

# --- Configuration and Initialization ---
load_dotenv()
genai.configure(api_key=os.getenv("API_KEY_GM"))
model = genai.GenerativeModel("gemini-2.5-flash")

# --- Initialize Timezone ---
ANTIPOLO_TZ = pytz.timezone('Asia/Manila')

# --- Database Configuration ---
DATABASE_PATH = "bank_transactions.db"

# --- Initialize database connection ---
def get_db_connection():
    """Establishes and returns a database connection."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        logger.debug("Database connection established.") 
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database at {DATABASE_PATH}: {e}", exc_info=True)
        raise

_cached_schema = None
def get_schema_string():
    """Retrieves and caches the database schema."""
    global _cached_schema
    if _cached_schema is not None:
        return _cached_schema
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        tables_info = []
        # Corrected: Use AND for multiple conditions in WHERE clause
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [t[0] for t in cursor.fetchall()]

        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            col_defs = ", ".join([f"{col[1]} {col[2]}" for col in columns])
            tables_info.append(f"{table}({col_defs})")
        schema_str = "\n".join(tables_info)
        _cached_schema = schema_str
        logger.info("Database schema retrieved and cached.")
        return schema_str
    except Exception as e:
        logger.critical(f"Failed to retrieve database schema: {e}. Application cannot function.", exc_info=True)
        raise
    finally:
        if conn:
            conn.close()

try:
    get_schema_string()
except Exception:
    logger.critical("Failed to load database schema on startup. Exiting.")
    exit(1)


# --- Holiday Data Fetching ---
_cached_ph_holidays = None
def get_philippine_holidays_cached():
    """Fetches and caches Philippine holidays for the current and next year."""
    global _cached_ph_holidays
    if _cached_ph_holidays is not None:
        return _cached_ph_holidays

    try:
        today = datetime.now()
        current_year = today.year
        next_year = today.year + 1

        ph_holidays = holidays.PH(years=[current_year, next_year])

        holiday_list = []
        for date, name in sorted(ph_holidays.items()):
            formatted_date = date.strftime('%B %d, %Y')
            holiday_list.append(f"{formatted_date}: {name}")
        _cached_ph_holidays = holiday_list
        logger.info("Philippine holidays fetched and cached.")
        return holiday_list
    except Exception as e:
        logger.error(f"Failed to fetch Philippine holidays: {e}", exc_info=True)
        _cached_ph_holidays = ["Error fetching holidays."]
        return _cached_ph_holidays

get_philippine_holidays_cached()


# --- Prompt Generation Function ---
def get_banking_assistant_prompt(schema_str, preferred_language, conversation_history_list=None, query_results_with_headers=None):
    """
    Constructs the prompt for the Gemini model based on the schema, preferred language,
    conversation history, and SQL query results.
    """
    ph_holidays = get_philippine_holidays_cached()
    holidays_str = "\n".join([f"- {h}" for h in ph_holidays])
    current_date_for_prompt = datetime.now(ANTIPOLO_TZ)

    language_instruction = ""
    examples_section = ""

    # Define common examples once
    common_example_sql = "SELECT SUM(COALESCE(Withdrawal_Amount, 0)) FROM bank_transactions WHERE LOWER(\"Transaction_Details\") LIKE '%service charge%';"
    common_example_nl_en = "You've spent a total of ₱500.00 on service charges. Is there anything else you'd like to check about your expenses?"
    common_example_nl_es = "Ha gastado un total de ₱500.00 en cargos por servicio. ¿Hay algo más que le gustaría revisar sobre sus gastos?"

    if preferred_language == "es":
        language_instruction = "Debes responder completamente en español latinoamericano. Todos los términos financieros, saludos y explicaciones deben estar en español latinoamericano."
        examples_section = f"""
Ejemplo (Inglés):

Usuario: ¿Cuánto gasté en cargos por servicio?

Respuesta:
{{
  "sql": "{common_example_sql}",
  "natural_language_response": "{common_example_nl_es}"
}}

Ejemplo (Español):

Usuario: ¿Cuánto gasté en cargos por servicio?

Respuesta:
{{
  "sql": "{common_example_sql}",
  "natural_language_response": "{common_example_nl_es}"
}}
"""
    else: # Default to English for 'en' or any other value
        language_instruction = "You must respond entirely in English. All financial terms, greetings, and explanations should be in English."
        examples_section = f"""
Example:

User: How much did I spend on service charges?

Response:
{{
  "sql": "{common_example_sql}",
  "natural_language_response": "{common_example_nl_en}"
}}
"""

    prompt_parts = [f"""
You are a friendly and intelligent banking assistant that helps users understand their financial activity by translating questions into SQL and providing clear, conversational answers.

{language_instruction}

Your expertise is with bank accounts of a bank, and you're familiar with Philippine banking habits, cities (including acronyms like QC, MKT, etc.), and common transaction types (e.g., service charges, deposits, ATM Withdrawal_Amount). Do not mention the bank name, but you must answer if the user asked the bank name.

Database schema:
{schema_str}

Current Date: {current_date_for_prompt.strftime('%B %d, %Y %I:%M:%S %p %Z')}

Philippine Holidays (for context, not for SQL queries unless explicitly asked about transactions on these dates):
{holidays_str}

Guidelines for SQL generation:
- **ONLY generate SELECT SQL queries.** Do NOT generate INSERT, UPDATE, DELETE, CREATE, DROP, or any other DDL/DML statements.
- SQL keywords (e.g., SELECT, FROM, WHERE, SUM) can be in uppercase or lowercase. For readability, lowercase is preferred.
- When you are greeted with a "hi", "hello", or any greetings, use a friendly, casual tone. Do not indicate the transactions yet or indicate that you are experiencing an error.
- For each user question, arrange the date in ascending order using `ORDER BY Date_Timestamp ASC`.
- When summing amounts like Deposit_Amount or Withdrawal_Amount, always use `COALESCE(column, 0)` to treat NULL as zero.
- Quote column names with spaces or special characters (e.g., "Transaction_Details") in SQL using double quotes. Ensure the casing of quoted column names (e.g., "Date_Timestamp", "Deposit_Amount") precisely matches the schema.
- **For general transaction listings (e.g., "Show me all transactions for X"), include the following columns for comprehensiveness: Date_Timestamp, Transaction_Details, Merchant_Counterparty_Name, Withdrawal_Amount, Deposit_Amount, Recurring_Existing_Balance.**
- When filtering for positive amounts (e.g., Withdrawal_Amount > 0) or specific values, always include `IS NOT NULL` condition if the column can contain NULLs and NULLs should not be considered in the filter (e.g., `WHERE Withdrawal_Amount IS NOT NULL AND Withdrawal_Amount > 0`).
- For transaction details, always use `LIKE` with appropriate wildcards (`%`) for partial matches (e.g., `WHERE "Transaction_Details" LIKE '%ATM WITHDRAWAL%'`). Use the exact string 'ATM WITHDRAWAL' for ATM transactions.
- For date filtering, use `strftime('%Y-%m-%d', Date_Timestamp)` for specific days, `strftime('%Y-%m', Date_Timestamp)` for months, and `strftime('%Y', Date_Timestamp)` for years.
- For summing transactions with similar merchant names, use `GROUP BY` on the relevant `Merchant_Counterparty_Name` column (or a substring if appropriate) and aggregate the amounts.

When replying, first provide the SQL query, and then, using the results of that query, generate a friendly, conversational, and emotionally aware response. Do NOT include the SQL query in your final response to the user.

Return your response as a JSON object with two keys:
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
            elif turn["role"] == "assistant" and "content" in turn: 
                # Only include the natural language content of previous assistant turns
                # It's important to be careful here: the current get_gemini_response
                # returns a dict with 'sql' and 'natural_language_response'.
                # For `conversation_history_list` to contain just the 'content' of assistant turns,
                # you need to ensure you're only appending the NL response to `messages`.
                # Let's adjust the way messages are stored slightly in the chat route.
                prompt_parts.append(f"Assistant: {turn['content']}")
        prompt_parts.append("----------------------------")

    if query_results_with_headers:
        headers = query_results_with_headers['headers']
        rows = query_results_with_headers['rows']

        if rows:
            formatted_results_for_prompt = [headers] + [[str(item) for item in row] for row in rows]
        else:
            formatted_results_for_prompt = []
        
        prompt_parts.append(f"\n--- SQL Query Results ---")
        prompt_parts.append(json.dumps(formatted_results_for_prompt, indent=2))
        prompt_parts.append("-------------------------")

    return "\n".join(prompt_parts)

# --- Gemini Interaction Function (used by both app and evaluation_module) ---
def get_gemini_response(user_question, model_obj, preferred_language, conversation_history_list=None, query_results_with_headers=None):
    """
    Sends a prompt to the Gemini model and parses its JSON response.
    """
    full_prompt = get_banking_assistant_prompt(get_schema_string(), preferred_language, conversation_history_list, query_results_with_headers)
    full_prompt_with_query = f"{full_prompt}\n\nUser: {user_question}\n\nAssistant Response:"

    logger.debug(f"Sending prompt to Gemini: {full_prompt_with_query[:500]}...")

    try:
        gemini_response = model_obj.generate_content(full_prompt_with_query)
        content = gemini_response.text.strip()
        logger.debug(f"Raw Gemini response received: {content[:500]}...")

        # Robust parsing for JSON block
        json_str = content
        if content.strip().startswith("```json"):
            json_str = content.strip()[len("```json"):].strip()
            if json_str.endswith("```"):
                json_str = json_str[:-len("```")].strip()
        elif content.strip().startswith("```"):
            json_str = content.strip()[len("```"):].strip()
            if json_str.endswith("```"):
                json_str = json_str[:-len("```")].strip()

        response_dict = json.loads(json_str)
        logger.info("Successfully parsed Gemini response into dictionary.")
        return response_dict
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from Gemini: {e}. Raw content: START>>>{content}<<<END", exc_info=True)
        # Attempt to recover an NL response if JSON fails but there's some text
        if "natural_language_response" in content:
            # This is a crude attempt; for production, a more robust regex might be needed
            try:
                nl_part = content.split('"natural_language_response": "')[1].split('"')[0]
                return {"error": f"JSON decode error: {e}", "natural_language_response": nl_part}
            except IndexError:
                pass
        return {"error": f"Error decoding JSON from Gemini: {e}", "raw_content": content}
    except Exception as e:
        logger.error(f"An error occurred during Gemini API call: {e}", exc_info=True)
        return {"error": f"An error occurred during Gemini API call: {e}"}

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main chat interface."""
    if 'messages' not in session:
        session['messages'] = []
        logger.info("Initialized new chat session (messages in Flask session).")
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles user chat queries, interacts with Gemini, and manages database queries."""
    user_query = request.json.get('query', '').strip()
    preferred_language = request.json.get('language', 'en')

    if not user_query:
        logger.warning("Received empty query from frontend.")
        return jsonify({"response": "Please enter a query." if preferred_language == "en" else "Por favor, introduce una consulta."}), 400

    logger.info(f"User Query ({preferred_language}): '{user_query}'")

    messages = session.get('messages', [])
    # Append user message for history before any processing
    messages.append({"role": "user", "content": user_query}) 

    exit_phrases_en = ["exit", "goodbye", "thank you", "thanks", "thats all", "that's all", "bye"]
    exit_phrases_es = ["salir", "adiós", "gracias", "eso es todo", "hasta luego"]
    greeting_phrases_en = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    greeting_phrases_es = ["hola", "buenos días", "buenas tardes", "buenas noches"]

    # --- Handle Exit Phrases ---
    if (preferred_language == "en" and any(phrase in user_query.lower() for phrase in exit_phrases_en)) or \
       (preferred_language == "es" and any(phrase in user_query.lower() for phrase in exit_phrases_es)):
        assistant_response_text = "¡Gracias por chatear! ¡Que tengas un gran día!" if preferred_language == "es" else "Thanks for chatting! Have a great day!"
        
        # Store only the NL content for the assistant's turn in history
        messages.append({"role": "assistant", "content": assistant_response_text}) # Removed 'sql' key for simple NL turns
        session['messages'] = messages
        logger.info(f"Chat ended with exit phrase. Assistant Response: '{assistant_response_text}'")
        return jsonify({"response": assistant_response_text, "sql_query": None})

    # --- Handle Greeting Phrases ---
    if (preferred_language == "en" and any(phrase in user_query.lower() for phrase in greeting_phrases_en)) or \
       (preferred_language == "es" and any(phrase in user_query.lower() for phrase in greeting_phrases_es)):
        assistant_response_text = "¡Hola! ¿Cómo puedo ayudarte hoy con tus finanzas?" if preferred_language == "es" else "Hi there! How can I help you with your finances today?"
        
        # Store only the NL content for the assistant's turn in history
        messages.append({"role": "assistant", "content": assistant_response_text}) # Removed 'sql' key
        session['messages'] = messages
        logger.info(f"Chat responded with greeting. Assistant Response: '{assistant_response_text}'")
        return jsonify({"response": assistant_response_text, "sql_query": None})

    # --- Proceed with SQL generation if not an exit or greeting ---
    sql_query = None
    assistant_response_text = ""

    # First call to Gemini for SQL generation
    response_for_sql = get_gemini_response(user_query, model, preferred_language, conversation_history_list=messages)

    if response_for_sql and "sql" in response_for_sql and response_for_sql["sql"]: # Ensure 'sql' key exists and is not empty
        sql_query = response_for_sql["sql"]
        logger.info(f"Generated SQL Query: '{sql_query}'")
        
        # CRITICAL SECURITY WARNING:
        # Direct execution of LLM-generated SQL carries a significant SQL Injection risk.
        # For a production system, you MUST implement robust SQL parsing, validation, and sanitization
        # to ensure only safe 'SELECT' queries are executed and malicious commands are blocked.
        # This demonstration code executes the SQL directly for evaluation purposes.
        logger.warning(f"SECURITY ALERT: Directly executing LLM-generated SQL: '{sql_query}'. This is dangerous in production without proper validation.")
        
    else:
        # Gemini failed to generate a valid SQL query
        assistant_response_text = "Lo siento, no pude generar una consulta SQL para esa solicitud. ¿Podrías reformularla?" if preferred_language == "es" else "I'm sorry, I couldn't generate a SQL query for that request. Could you please rephrase it?"
        
        # Store only the NL content for assistant's turn in history
        messages.append({"role": "assistant", "content": assistant_response_text}) # Removed 'sql' key
        session['messages'] = messages
        logger.warning(f"Gemini failed to generate SQL for user query: '{user_query}'. Assistant response: '{assistant_response_text}'")
        return jsonify({"response": assistant_response_text, "sql_query": None})

    conn = None
    query_results_with_headers = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql_query)
        query_results = cursor.fetchall()
        
        column_headers = [description[0] for description in cursor.description]
        query_results_with_headers = {"headers": column_headers, "rows": [list(row) for row in query_results]}
        logger.info(f"SQL Query Executed Successfully. Results count: {len(query_results)} rows.")

        # Second call to Gemini for NL response, now with SQL results
        final_response_dict = get_gemini_response(
            user_query, model,
            preferred_language,
            conversation_history_list=messages, # Pass updated messages for context
            query_results_with_headers=query_results_with_headers
        )

        if final_response_dict and "natural_language_response" in final_response_dict:
            assistant_response_text = final_response_dict['natural_language_response']
            logger.info(f"Final Assistant Response (NL): '{assistant_response_text}'")
        else:
            assistant_response_text = "Ejecuté la consulta, pero tuve problemas para formar una respuesta clara. Por favor, inténtalo de nuevo." if preferred_language == "es" else "I executed the query, but I had trouble forming a clear response. Please try again."
            logger.error(f"Gemini failed to generate final NL response after query execution. Assistant response: '{assistant_response_text}'")
            # If NL generation failed, we might not want to show the SQL query to the user
            sql_query = None 

    except sqlite3.Error as e:
        assistant_response_text = (
            f"Tuve un problema al procesar esa solicitud: Error de base de datos. ¿Podrías reformular tu pregunta?"
            if preferred_language == "es" else
            f"I ran into an issue processing that request: Database error. Could you try rephrasing your question?"
        )
        logger.error(f"Database error during SQL execution for query '{sql_query}': {e}", exc_info=True)
        sql_query = None # Clear SQL query if execution failed
    except Exception as e:
        assistant_response_text = (
            f"Ocurrió un error inesperado. Por favor, inténtalo de nuevo o reformula tu pregunta."
            if preferred_language == "es" else
            f"An unexpected error occurred. Please try again or rephrase your question."
        )
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
        sql_query = None # Clear SQL query on unexpected error
    finally:
        if conn:
            conn.close()

    # Append the assistant's final response (NL + potentially SQL) to session messages
    # Only store the NL content for the history for the *prompt*, but keep SQL in the jsonify for the UI.
    messages.append({"role": "assistant", "content": assistant_response_text, "sql": sql_query})
    session['messages'] = messages

    return jsonify({"response": assistant_response_text, "sql_query": sql_query})

# --- EVALUATION ROUTE ---
@app.route('/evaluate', methods=['GET'])
def evaluate():
    """
    Triggers the evaluation process for the banking assistant.
    This route is for internal testing/evaluation, not for public use.
    """
    logger.info("Evaluation route triggered.")
    
    if not os.getenv("API_KEY_GM"):
        error_msg = "Cannot run evaluation: Google Gemini API key not configured."
        logger.error(error_msg)
        return jsonify({"message": error_msg, "status": "error"}), 500

    try:
        evaluation_results_output = evaluation_module.run_evaluation(
            model_obj=model,
            get_gemini_response_func=get_gemini_response,
            get_db_connection_func=get_db_connection,
            get_schema_string_func=get_schema_string,
            logger=logger,
            evaluation_output_dir=EVALUATION_OUTPUT_DIR,
            antipolo_tz=ANTIPOLO_TZ
        )
        logger.info("Evaluation process completed successfully.")
        return jsonify(evaluation_results_output)
    except Exception as e:
        error_msg = f"An error occurred during the evaluation process: {e}"
        logger.critical(error_msg, exc_info=True)
        return jsonify({"message": error_msg, "status": "error"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)