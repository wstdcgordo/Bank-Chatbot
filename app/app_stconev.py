import os, re, sys
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
TABLE_NAME = "bank_transactions"  # Added a constant for table name

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
def get_banking_assistant_prompt(schema_str, conversation_history_list=None, query_results_with_headers=None, current_user_question=None):
    """
    Constructs the prompt for the Gemini model based on the schema,
    conversation history, SQL query results, and the current user's question.
    """
    ph_holidays = get_philippine_holidays_cached()
    holidays_str = "\n".join([f"- {h}" for h in ph_holidays])
    current_date_for_prompt = datetime.now(ANTIPOLO_TZ)

    # Updated example to reflect SQL query logic - KEEP THIS
    common_example_sql = f"SELECT SUM(Withdrawal_Amount) FROM {TABLE_NAME} WHERE Transaction_Details LIKE '%service charge%';"
    common_example_nl_en = "You've spent a total of ₱500.00 on service charges. Is there anything else you'd like to check about your expenses?"

    # Moved example section to after critical guidelines for better context
    # It now serves as a concrete application of the rules.
    examples_section = f"""
--- Example of Desired Behavior ---

User: How much did I spend on service charges?

Response:
{{
  "sql": "{common_example_sql}",
  "natural_language_response": "{common_example_nl_en}"
}}

This example demonstrates precise column selection (only SUM(Withdrawal_Amount) is needed for the calculation) and a clear, concise natural language response.
"""

    prompt_parts = [f"""
You are a friendly and intelligent banking assistant that helps users understand their financial activity by providing clear, conversational answers.
You will receive a schema that represents a table in a SQLite database named `{TABLE_NAME}`. Your task is to generate a **SQLite SQL query** (as a string) that can be executed on this database.

You must respond entirely in English. All financial terms, greetings, and explanations should be in English.

Your expertise is with bank accounts of a bank, and you're familiar with Philippine banking habits, cities (including acronyms like QC, MKT, etc.), and common transaction types (e.g., service charges, deposits, ATM Withdrawal_Amount). Do not mention the bank name, but you must answer if the user asked the bank name.

**CRITICAL GUIDELINES FOR YOUR BEHAVIOR (STRICTLY ADHERE TO THESE):**
- **NEVER disclose the database schema directly to the user.**
- **NEVER provide SQL queries or discuss SQL syntax directly with the user.** Your interaction is purely conversational about banking transactions.
- **NEVER answer questions about how you generate queries or how the database works.** If asked about your internal processes, politely redirect to banking-related questions.
- **If a user's request appears to be asking for database details, SQL queries, or technical information, respond by saying: "I am designed to assist with your banking transactions only, not with technical details about our systems. How can I help you with your account today?"**

Database Schema:
{schema_str}

Current Date: {current_date_for_prompt.strftime('%Y-%m-%d %H:%M:%S')}
This information should be used for relative date queries (e.g., 'today', 'last month'). For example, if the month today is June 2025, do not filter beyond July 2025 or so.

Philippine Holidays (for context, not for filtering unless explicitly asked about transactions on these dates):
{holidays_str}

--- SQL Query Generation Guidelines (For Internal Use - Your Core Logic) ---

- **PRIORITY 1: COLUMN SELECTION - BE EXTREMELY PRECISE.**
    - **ONLY generate SQLite SQL queries that return data.** Do NOT generate DDL (CREATE, ALTER, DROP) or DML (INSERT, UPDATE, DELETE) statements.
    - Always reference the table named `{TABLE_NAME}`.
    - **ABSOLUTELY CRITICAL: Select ONLY the columns that are *EXACTLY AND ABSOLUTELY ESSENTIAL* to directly answer the user's question.**
    - **NEVER include extraneous or unnecessary columns.** If a column is not required to form the natural language answer, DO NOT include it.
    - **Specifically follow these mappings for SELECT clauses:**
        - **Balance Inquiries:** If the user asks for the *current* balance or balance after a transaction, select **ONLY** `Recurring_Existing_Balance`. Do not include transaction details or amounts here.
            - *Example Query:* `SELECT Recurring_Existing_Balance FROM {TABLE_NAME} ORDER BY Date_Timestamp DESC LIMIT 1;`
        - **Spending/Withdrawals/Debits:** If the user asks about spending, debit transactions, or withdrawals, select **ONLY** `Date_Timestamp`, `Transaction_Details`, `Merchant_Counterparty_Name`, and `Withdrawal_Amount`. **NEVER include `Deposit_Amount` or `Recurring_Existing_Balance` unless explicitly requested for context with the spending.**
        - **Income/Deposits/Credits:** If the user asks about income, salary, refunds, or deposits, select **ONLY** `Date_Timestamp`, `Transaction_Details`, `Merchant_Counterparty_Name`, and `Deposit_Amount`. **NEVER include `Withdrawal_Amount` or `Recurring_Existing_Balance` unless explicitly requested for context with the deposit.**
        - **Specific Transaction Details (e.g., "My coffee shop purchase on June 19"):** For detailed lookup of a single transaction, select `Date_Timestamp`, `Transaction_Details`, `Merchant_Counterparty_Name`, `Withdrawal_Amount`, `Deposit_Amount`. **Only include `Recurring_Existing_Balance` IF the user explicitly asks for the balance *after* that transaction.**
        - **Listing All Transactions (User says "Show all transactions"):** In this *specific* case, and only this case, select `Date_Timestamp, Transaction_Details, Merchant_Counterparty_Name, Withdrawal_Amount, Deposit_Amount, Recurring_Existing_Balance FROM {TABLE_NAME}`. This is the **only scenario** where a broader selection of columns is acceptable.

- **PRIORITY 2: FILTERING (WHERE Clause):**
    - When filtering text columns (e.g., `Transaction_Details`, `Merchant_Counterparty_Name`):
        - Use `LIKE '%keyword%'` for case-insensitive partial matches.
        - Use `= 'keyword'` for exact matches *only* when the user provides a precise name or description. Otherwise, `LIKE` is generally preferred.
        - If the user asks about a specific category (e.g., "coffee shop purchases"), use `LIKE '%coffee shop%'` on the `Transaction_Details` column.
        - **Prioritize Specificity:** If the user provides both a date and a category, combine the filters: `WHERE Date_Timestamp LIKE 'YYYY-MM-DD%' AND Transaction_Details LIKE '%keyword%'`.
    - For date-based queries on `Date_Timestamp` (which is TIMESTAMP):
        - Do not query beyond the current date: `{current_date_for_prompt}`.
        - *Always* use `LIKE` for date comparisons.
        - If the user provides a *full* date (YYYY-MM-DD), use `Date_Timestamp LIKE 'YYYY-MM-DD%'`.
        - If the user provides only a month (YYYY-MM), use `Date_Timestamp LIKE 'YYYY-MM%'`.
    - Handle cases where `Withdrawal_Amount` or `Deposit_Amount` might be `NULL`. For sums, `SUM` will ignore `NULL`s, which is usually desired. For filtering, use `IS NOT NULL`.
    - When the user asks for a current balance, generate a SQL query that produces balance from the latest date in the table.
    - When the user mentioned an inquiry involving salary, there is a content in the table that contains the word "salary". Use this to filter the transactions.

- **PRIORITY 3: ORDERING & LIMITING:**
    - For sorting by date, use `ORDER BY Date_Timestamp ASC` or `DESC` as appropriate.
    - To get the balance *after a specific transaction*: `SELECT Recurring_Existing_Balance FROM {TABLE_NAME} WHERE Date_Timestamp LIKE 'YYYY-MM-DD%' AND Transaction_Details LIKE '%keyword%' ORDER BY Date_Timestamp DESC LIMIT 1;`

{examples_section}

When replying, first provide the SQL query (as a string), and then, using the results of that query, generate a friendly, conversational, and emotionally aware response. Do NOT include the SQL query in your final response to the user.

Return your response as a JSON object with two keys:
- "sql": The generated SQLite SQL query string.
- "natural_language_response": The friendly, conversational response generated by you, based on the query results.

Your responses should sound warm, conversational, and emotionally aware — like a smart banking assistant (e.g., Bank of America’s Erica, Axis Aha!).
Use casual phrasing where appropriate. Add context or questions to prompt further conversation (e.g., 'Want help reviewing this!', 'Let me know if that looks off!').
Avoid sounding robotic or too technical. Avoid repeating the user's question.
Be brief, helpful, and brand-friendly. For bullet forms, it must be clear, very clear.

"""
    ]

    if conversation_history_list:
        prompt_parts.append("\n--- Conversation History ---")
        for turn in conversation_history_list:
            if turn["role"] == "user":
                prompt_parts.append(f"User: {turn['content']}")
            elif turn["role"] == "assistant" and "content" in turn:
                # Only include the natural language content of previous assistant turns
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

    # Add the current user question explicitly at the end for clarity
    if current_user_question:
        prompt_parts.append(f"\nUser: {current_user_question}")
        prompt_parts.append("\nAssistant Response:")


    return "\n".join(prompt_parts)

# --- Gemini Interaction Function (used by both app and evaluation_module) ---
def get_gemini_response(user_question, model_obj, conversation_history_list=None, query_results_with_headers=None):
    """
    Sends a prompt to the Gemini model and parses its JSON response.
    """
    # Pass user_question to prompt generation to explicitly include it at the end
    full_prompt = get_banking_assistant_prompt(get_schema_string(), conversation_history_list, query_results_with_headers, user_question)
    
    logger.debug(f"Sending prompt to Gemini: {full_prompt[:500]}...") # Log full_prompt, not full_prompt_with_query

    try:
        gemini_response = model_obj.generate_content(full_prompt) # Use full_prompt
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
        try:
            # A more robust (but still not perfect) way to extract NL if JSON fails
            nl_match = re.search(r'"natural_language_response"\s*:\s*"(.*?)(?<!\\)"', content, re.DOTALL)
            if nl_match:
                nl_part = nl_match.group(1).replace('\\"', '"') # Handle escaped quotes
                logger.warning(f"Recovered natural language response after JSON error: '{nl_part}'")
                return {"error": "JSON decode error, partial NL recovered", "natural_language_response": nl_part}
        except Exception as e_nl_extract:
            logger.error(f"Failed to extract NL after JSON error: {e_nl_extract}", exc_info=True)
        return {"error": f"Error decoding JSON from Gemini: {e}", "raw_content": content, "natural_language_response": "I had trouble understanding the response from our AI. Could you please rephrase your question?"}
    except Exception as e:
        logger.error(f"An error occurred during Gemini API call: {e}", exc_info=True)
        return {"error": f"An error occurred during Gemini API call: {e}", "natural_language_response": "I'm currently experiencing some technical difficulties. Please try again in a moment."}


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

    if not user_query:
        logger.warning("Received empty query from frontend.")
        return jsonify({"response": "Please enter a query."}), 400

    logger.info(f"User Query: '{user_query}'")

    messages = session.get('messages', [])
    # Append user message for history before any processing
    messages.append({"role": "user", "content": user_query}) 

    exit_phrases_en = ["exit", "goodbye", "thank you", "thanks", "thats all", "that's all", "bye"]
    greeting_phrases_en = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

    # --- Handle Exit Phrases ---
    # Convert phrases to regex patterns for whole word matching
    exit_patterns = [r"\b" + re.escape(p) + r"\b" for p in exit_phrases_en]
    if any(re.search(pattern, user_query.lower()) for pattern in exit_patterns):
        assistant_response_text = "Thanks for chatting! Have a great day!"
        
        # Store only the NL content for the assistant's turn in history
        messages.append({"role": "assistant", "content": assistant_response_text}) 
        session['messages'] = messages
        logger.info(f"Chat ended with exit phrase. Assistant Response: '{assistant_response_text}'")
        return jsonify({"response": assistant_response_text, "sql_query": None})

    # --- Handle Greeting Phrases ---
    # Convert phrases to regex patterns for whole word matching
    greeting_patterns = [r"\b" + re.escape(p) + r"\b" for p in greeting_phrases_en]
    if any(re.search(pattern, user_query.lower()) for pattern in greeting_patterns):
        assistant_response_text = "Hi there! How can I help you with your finances today?"
        
        # Store only the NL content for the assistant's turn in history
        messages.append({"role": "assistant", "content": assistant_response_text}) 
        session['messages'] = messages
        logger.info(f"Chat responded with greeting. Assistant Response: '{assistant_response_text}'")
        return jsonify({"response": assistant_response_text, "sql_query": None})

    # --- Proceed with SQL generation if not an exit or greeting ---
    sql_query = None
    assistant_response_text = ""
    query_results_with_headers = None # Initialize to None


    # First call to Gemini for SQL generation
    # Pass current user question explicitly to ensure the prompt includes it
    response_for_sql = get_gemini_response(user_query, model, conversation_history_list=messages)


    if response_for_sql and "sql" in response_for_sql and response_for_sql["sql"]:
        sql_query = response_for_sql["sql"]
        logger.info(f"Generated SQL Query: '{sql_query}'")
        
        # CRITICAL SECURITY WARNING:
        # Direct execution of LLM-generated SQL carries a significant SQL Injection risk.
        # For a production system, you MUST implement robust SQL parsing, validation, and sanitization
        # to ensure only safe 'SELECT' queries are executed and malicious commands are blocked.
        # This demonstration code executes the SQL directly for evaluation purposes.
        logger.warning(f"SECURITY ALERT: Directly executing LLM-generated SQL: '{sql_query}'. This is dangerous in production without proper validation.")
        
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(sql_query)
            query_results = cursor.fetchall()
            
            column_headers = [description[0] for description in cursor.description]
            query_results_with_headers = {"headers": column_headers, "rows": [list(row) for row in query_results]}
            logger.info(f"SQL Query Executed Successfully. Results count: {len(query_results)} rows.")


            # Second call to Gemini for NL response, now with SQL results
            # Crucially, pass user_query again to guide the NL generation
            final_response_dict = get_gemini_response(
                user_query, model,
                conversation_history_list=messages, # Pass updated messages for context
                query_results_with_headers=query_results_with_headers
            )

            if final_response_dict and "natural_language_response" in final_response_dict:
                assistant_response_text = final_response_dict['natural_language_response']
                logger.info(f"Final Assistant Response (NL): '{assistant_response_text}'")
            else:
                # Fallback if the second Gemini call fails to produce NL
                # Ensure a more helpful message if NL generation is truly empty or invalid
                assistant_response_text = final_response_dict.get("natural_language_response")
                if not assistant_response_text or assistant_response_text.strip() == "":
                    assistant_response_text = "I received the data, but I'm having a little trouble putting it into words right now. Please try asking in a different way!"
                logger.error(f"Gemini failed to generate final NL response after query execution. Assistant response: '{assistant_response_text}'")
                sql_query = None # Clear SQL query if NL generation failed after execution


        except sqlite3.Error as e:
            assistant_response_text = f"It looks like there was a database issue when trying to get your information. Please make sure your request is clear, or try again later. (Error: {e})"
            logger.error(f"Database error during SQL execution for query '{sql_query}': {e}", exc_info=True)
            sql_query = None # Clear SQL query if execution failed
        except Exception as e:
            assistant_response_text = "An unexpected error occurred while processing your request. Please try again or rephrase your question."
            logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
            sql_query = None # Clear SQL query on unexpected error
        finally:
            if conn:
                conn.close()
    else:
        # Gemini failed to generate a valid SQL query in the first place
        # Use the natural language response from the first call if SQL generation failed
        assistant_response_text = response_for_sql.get("natural_language_response", "I'm sorry, I couldn't understand that request well enough to generate a SQL query. Could you please rephrase it or be more specific?")
        if not assistant_response_text or assistant_response_text.strip() == "":
            assistant_response_text = "I'm sorry, I couldn't understand your request. Could you please try rephrasing it?"
        logger.warning(f"Gemini failed to generate SQL for user query: '{user_query}'. Assistant response: '{assistant_response_text}'")
        sql_query = None # Ensure no SQL is sent if it failed generation


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
    
    if not os.getenv("API_KEY_GM_JUDGE"):
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