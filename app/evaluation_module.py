import os
import json
import sqlite3
import csv
from datetime import datetime, timedelta
import logging
import re
from decimal import Decimal, InvalidOperation # For robust numerical comparison
import pytz

antipolo_tz = pytz.timezone('Asia/Manila')

def get_last_week_date_range(current_evaluation_time: datetime):
    """
    Calculates the start (Monday 00:00:00) and end (Sunday 23:59:59)
    of the *previous* full calendar week, relative to the provided time.
    """
    # Ensure the input time is timezone-aware
    if current_evaluation_time.tzinfo is None or current_evaluation_time.tzinfo.utcoffset(current_evaluation_time) is None:
        raise ValueError("current_evaluation_time must be timezone-aware")

    # 1. Get the start of the current day (midnight) in the specified timezone
    today_start = current_evaluation_time.replace(hour=0, minute=0, second=0, microsecond=0)

    # 2. Find the start of the *current calendar week* (Monday)
    # weekday() returns 0 for Monday, 6 for Sunday
    start_of_current_week = today_start - timedelta(days=today_start.weekday())

    # 3. Calculate the start of "last week" (Monday of the previous week)
    start_of_last_week = start_of_current_week - timedelta(days=7)

    # 4. Calculate the end of "last week" (Sunday of the previous week, which is one day before the start of the current week)
    end_of_last_week = start_of_current_week - timedelta(days=1)
    end_of_last_week = end_of_last_week.replace(hour=23, minute=59, second=59, microsecond=999999) # End of day

    return (start_of_last_week, end_of_last_week)

start_date_obj, end_date_obj = get_last_week_date_range(datetime.now(antipolo_tz))
start_date_str = start_date_obj.strftime('%Y-%m-%d %H:%M:%S')
end_date_str = end_date_obj.strftime('%Y-%m-%d %H:%M:%S') # Or %Y-%m-%d %H:%M:%S.%f if you need microseconds

# Configure logging for the evaluation module
eval_logger = logging.getLogger(__name__)
eval_logger.setLevel(logging.INFO)

if not eval_logger.handlers:
    eval_log_dir = "logs"
    os.makedirs(eval_log_dir, exist_ok=True)
    eval_file_handler = logging.FileHandler(os.path.join(eval_log_dir, "evaluation.log"))
    eval_file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    eval_file_handler.setFormatter(formatter)
    eval_logger.addHandler(eval_file_handler)
    # Also add console handler for immediate feedback during development
    eval_console_handler = logging.StreamHandler()
    eval_console_handler.setLevel(logging.INFO)
    eval_console_handler.setFormatter(formatter)
    eval_logger.addHandler(eval_console_handler)

eval_logger.info("Evaluation module loaded.")

MOCK_TABLE_NAME = "bank_transactions"

# --- Test Dataset for Evaluation ---
TEST_CASES = [
    {
        "id": "1",
        "question": "What is my current balance?",
        "language": "en",
        "query_type": "current_balance",
        "ground_truth_sql": f"SELECT COALESCE(Recurring_Existing_Balance, 0) FROM {MOCK_TABLE_NAME} ORDER BY Date_Timestamp DESC LIMIT 1;",
        "expected_nl_response": "Your current balance is ₱[amount]."
    },
    {
        "id": "2",
        "question": "Show me my last transaction.",
        "language": "en",
        "query_type": "last_transaction",
        "ground_truth_sql": f"SELECT Date_Timestamp, \"Transaction_Details\", COALESCE(Withdrawal_Amount, Deposit_Amount) as Amount, Recurring_Existing_Balance FROM {MOCK_TABLE_NAME} WHERE Date_Timestamp = (SELECT MAX(Date_Timestamp) FROM {MOCK_TABLE_NAME});",
        "expected_nl_response": "Your last transaction was on [date] for [details] amounting to ₱[amount]. Your balance after this was ₱[balance]."
    },
    {
        "id": "3",
        "question": "Did I withdraw from an ATM last week?",
        "language": "en",
        "query_type": "atm_withdrawal_last_week",
        # Assuming evaluation is run on/around June 25, 2025. Last week would be June 16-22.
        # Adjusted: Last week (Sunday to Saturday) prior to current week. If today is Wed 25th, last week was June 15-21
        # Let's adjust to be June 17-23 (Tuesday to Monday), based on your sample's fixed date.
        "ground_truth_sql": f"SELECT COUNT(*) FROM bank_transactions WHERE \"Transaction_Details\" LIKE '%ATM Withdrawal%' AND Date_Timestamp BETWEEN '{start_date_str}' AND '{end_date_str}' AND Withdrawal_Amount > 0;",
        "expected_nl_response": "Yes, you had [number] ATM withdrawals last week." # Or "No, you did not have any ATM withdrawals last week."
    },
    {
        "id": "4",
        "question": "How much did I deposit last Christmas?",
        "language": "en",
        "query_type": "deposit_last_christmas",
        "ground_truth_sql": f"SELECT SUM(COALESCE(Deposit_Amount, 0)) FROM {MOCK_TABLE_NAME} WHERE strftime('%Y-%m-%d', Date_Timestamp) = '2024-12-25' AND Deposit_Amount > 0;",
        "expected_nl_response": "Last Christmas, you deposited a total of ₱[amount]."
    },
    {
        "id": "5",
        "question": "How many times did I withdraw from BPI?",
        "language": "en",
        "query_type": "bpi_withdrawal_count",
        "ground_truth_sql": f"SELECT COUNT(*) FROM {MOCK_TABLE_NAME} WHERE \"Merchant_Counterparty_Name\" LIKE '%BPI%' AND Withdrawal_Amount > 0;",
        "expected_nl_response": "You withdrew from BPI [number] times."
    }
]

# --- Helper for Comparing Query Results (for Execution Accuracy) ---
def normalize_value(value):
    """Normalizes values for robust comparison."""
    if value is None:
        return "NULL_VALUE"
    if isinstance(value, (int, float)):
        try:
            # Convert to Decimal for precise comparison, then to float
            # or round to a fixed number of decimal places for comparison
            return float(Decimal(str(value)).quantize(Decimal('0.01'))) # Round to 2 decimal places
        except InvalidOperation:
            pass # Fallback to string if not a valid number
    return str(value).strip().lower() # Consistent string conversion


def compare_query_results(result1_rows, result2_rows, column_order_independent=True):
    """
    Compares two sets of query results for semantic equivalence.
    Normalizes values to handle type and precision differences.
    """
    if len(result1_rows) != len(result2_rows):
        return False, f"Row count mismatch: {len(result1_rows)} vs {len(result2_rows)}"

    canonical_result1 = []
    for row in result1_rows:
        normalized_row = tuple(normalize_value(col) for col in row)
        canonical_result1.append(frozenset(normalized_row) if column_order_independent else normalized_row)
    
    canonical_result2 = []
    for row in result2_rows:
        normalized_row = tuple(normalize_value(col) for col in row)
        canonical_result2.append(frozenset(normalized_row) if column_order_independent else normalized_row)

    # Sort canonical results for consistent comparison, especially if row order might vary
    # Sorting by string representation of the tuple/frozenset
    canonical_result1_sorted = sorted(canonical_result1, key=lambda x: str(x))
    canonical_result2_sorted = sorted(canonical_result2, key=lambda x: str(x))

    if canonical_result1_sorted == canonical_result2_sorted:
        return True, "Results match"
    else:
        diff_str = "Result mismatch detected. First differing rows (max 5):\n"
        diff_count = 0
        for i in range(len(canonical_result1_sorted)):
            if canonical_result1_sorted[i] != canonical_result2_sorted[i]:
                diff_str += f"  GT: {canonical_result1_sorted[i]}\n  Gen: {canonical_result2_sorted[i]}\n"
                diff_count += 1
                if diff_count >= 5: break
        return False, diff_str


# --- LLM Judge Prompt Helper ---
def get_llm_judge_prompt(user_question, generated_sql, ground_truth_sql, generated_nl_response, expected_nl_response, query_results_with_headers, schema_str):
    """Constructs the prompt for the LLM Judge to evaluate SQL and NL responses."""
    results_str = "N/A (No results or execution failed)"
    if query_results_with_headers and query_results_with_headers.get('headers') and query_results_with_headers.get('rows') is not None:
        headers, rows = query_results_with_headers['headers'], query_results_with_headers['rows']
        
        # Limit rows for prompt to avoid exceeding token limits for LLM judge
        display_rows = rows[:5] + (['... (truncated)'] if len(rows) > 5 else [])

        table_lines = [f"| {' | '.join(headers)} |", f"|{'-' * (sum(len(h) for h in headers) + 3 * len(headers) - 1)}|"]
        table_lines.extend(f"| {' | '.join(str(col) for col in row)} |" for row in display_rows)
        results_str = "\n".join(table_lines)

    # Ensure SQL strings are not None before formatting
    generated_sql_display = generated_sql if generated_sql is not None else "N/A (No SQL generated)"
    ground_truth_sql_display = ground_truth_sql if ground_truth_sql is not None else "N/A (No SQL expected/provided)"

    return f"""
You are an expert evaluator for a banking chatbot. Critically assess the generated SQL query and natural language response based on the user's question and schema.

**Instructions:**
1.  **Evaluate Generated SQL vs. Ground Truth SQL:**
    * **Semantic Equivalence:** Does `Generated SQL` produce the exact same result set as `Ground Truth SQL` given `User Question` and `Database Schema`? Consider if it's semantically identical even if syntax differs. If `Ground Truth SQL` is N/A, then the `Generated SQL` should also be N/A or empty.
    * **SQL Correctness Score:** Rate 1 (completely wrong or generated when not expected) to 5 (perfectly equivalent or correctly not generated).
2.  **Evaluate Natural Language Response:**
    * **Accuracy & Helpfulness:** Is `Generated Natural Language Response` accurate, helpful, conversational, and correctly answers `User Question` based on `SQL Query Results` (if applicable)? Compare to `Expected Natural Language Response`.
    * **NL Accuracy Score:** Rate 1 (completely inaccurate) to 5 (fully accurate).
    * **NL Helpfulness Score:** Rate 1 (not helpful) to 5 (very helpful and conversational).
3.  **Provide concise explanations for all scores.**
4.  **Output JSON format.**

**Input Data:**
User Question: {user_question}
Database Schema:\n{schema_str}
Ground Truth SQL: {ground_truth_sql_display}
Generated SQL: {generated_sql_display}
SQL Query Results (from Generated SQL):\n{results_str}
Generated Natural Language Response: {generated_nl_response}
Expected Natural Language Response (for reference): {expected_nl_response if expected_nl_response else "Not provided"}

**Output Format (JSON):**
{{
  "sql_correctness_score": [1-5],
  "sql_equivalence_verdict": "Yes" | "No" | "N/A",
  "sql_explanation": "Brief explanation for SQL verdict.",
  "nl_accuracy_score": [1-5],
  "nl_helpfulness_score": [1-5],
  "nl_explanation": "Brief explanation for NL scores and quality.",
  "overall_comment": "High-level comments."
}}
"""

# --- Function to Get LLM Judge's Response ---
def get_llm_judge_response(model_obj, prompt_text, logger):
    """Sends evaluation prompt to LLM Judge model and parses JSON response."""
    try:
        # Increase safety settings if the judge is failing due to content moderation
        generation_config = {
            "temperature": 0.1, # Keep temperature low for consistent evaluation
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        response = model_obj.generate_content(
            prompt_text,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        content = response.text.strip()
        logger.debug(f"Raw LLM Judge response: {content}...")
        logger.debug(f"Partial LLM Judge response for brevity: {content[:500]}...")

        # Robust JSON parsing from previous fix
        json_str = content
        if content.strip().startswith("```json"):
            json_str = content.strip()[len("```json"):].strip()
            if json_str.endswith("```"):
                json_str = json_str[:-len("```")].strip()
        elif content.strip().startswith("```"):
            json_str = content.strip()[len("```"):].strip()
            if json_str.endswith("```"):
                json_str = json_str[:-len("```")].strip()

        return json.loads(json_str)
    
    except json.JSONDecodeError as e:
        logger.error(f"LLM Judge JSON decoding error: {e}. Raw content: {content}", exc_info=True)
        return {"error": f"JSON decoding error: {e}", "raw_content": content}
    except Exception as e:
        logger.error(f"LLM Judge unexpected error during content generation: {e}", exc_info=True)
        return {"error": f"Unexpected error: {e}. Check API key, quotas, or model response blocking."}


# --- Evaluation Function for a Single Test Case ---
def evaluate_test_case(test_case: dict, model_obj, get_gemini_response_func, get_db_connection_func, get_schema_string_func, logger):
    """Evaluates a single test case."""
    eval_results = {
        "id": test_case["id"], "question": test_case["question"], "language": test_case["language"],
        "query_type": test_case.get("query_type", "unknown"), "generated_sql": None,
        "ground_truth_sql": test_case["ground_truth_sql"],
        "sql_execution_success": False, "ground_truth_execution_success": False, "sql_execution_match": False,
        "generated_nl_response": None, "expected_nl_response": test_case.get("expected_nl_response"),
        "llm_judge_verdict": None, "errors": []
    }

    generated_sql_results_with_headers = None
    ground_truth_sql_results_rows = []
    
    conn = None

    try:
        # 1. Get Generated SQL and initial NL (if applicable) from the main bot
        # Pass conversation_history_list as an empty list for independent evaluation if not meant to be multi-turn
        # For evaluation, we generally want each turn to be independent unless testing multi-turn specifically.
        # However, your prompt uses conversation history, so we should mock it or pass empty for first turn.
        # For simplicity, let's pass an empty list, assuming evaluation focuses on single-turn accuracy for now.
        # If testing multi-turn, you'd build a specific history for each test case.
        
        # The main app's get_gemini_response requires conversation_history_list
        # For initial SQL generation, we don't have prior turns, so pass an empty list
        response_from_bot_sql_phase = get_gemini_response_func(
            test_case["question"], model_obj, conversation_history_list=[] # Empty history for fresh query
        )

        if response_from_bot_sql_phase and "sql" in response_from_bot_sql_phase:
            eval_results["generated_sql"] = response_from_bot_sql_phase["sql"]
        # Even if SQL is None (e.g., greeting), we should still capture NL
        if response_from_bot_sql_phase and "natural_language_response" in response_from_bot_sql_phase:
            # For non-SQL queries (like greetings), the first call directly provides the NL
            eval_results["generated_nl_response"] = response_from_bot_sql_phase["natural_language_response"]
            logger.debug(f"Test Case {test_case['id']}: Initial NL response captured from first bot call.")
        else:
             eval_results["errors"].append(f"Bot failed to provide initial response/SQL in first call: {response_from_bot_sql_phase.get('error', 'Unknown error')}")
             logger.error(f"Test Case {test_case['id']}: Bot failed initial response generation.")
             # Continue to judge even if initial response failed, to let judge assess quality of "error" response

        # 2. Execute Ground Truth and Generated SQL queries (if SQL is expected)
        if test_case["ground_truth_sql"] is not None: # Only execute if SQL is expected for this test case
            try:
                conn = get_db_connection_func()
                cursor = conn.cursor()

                try:
                    cursor.execute(test_case["ground_truth_sql"])
                    ground_truth_sql_results_rows = cursor.fetchall()
                    eval_results["ground_truth_execution_success"] = True
                    logger.debug(f"Test Case {test_case['id']}: GT SQL executed.")
                except sqlite3.Error as e:
                    eval_results["errors"].append(f"GT SQL execution error: {e} for query: {test_case['ground_truth_sql']}")
                    eval_results["ground_truth_execution_success"] = False
                    logger.error(f"Test Case {test_case['id']}: GT SQL error: {e}")

                if eval_results["generated_sql"]: # Only attempt execution if generated SQL exists
                    try:
                        cursor.execute(eval_results["generated_sql"])
                        generated_sql_results_rows = cursor.fetchall()
                        eval_results["sql_execution_success"] = True
                        column_headers = [desc[0] for desc in cursor.description]
                        generated_sql_results_with_headers = {"headers": column_headers, "rows": [list(row) for row in generated_sql_results_rows]}
                        logger.debug(f"Test Case {test_case['id']}: Generated SQL executed.")
                    except sqlite3.Error as e:
                        eval_results["errors"].append(f"Generated SQL execution error: {e} for query: {eval_results['generated_sql']}")
                        eval_results["sql_execution_success"] = False
                        logger.error(f"Test Case {test_case['id']}: Generated SQL error: {e}")
                else:
                    eval_results["errors"].append("Generated SQL is None (as expected for greetings or generation failure), skipping execution.")
                    eval_results["sql_execution_success"] = False # Explicitly set to false if no SQL to execute

                # Compare results only if both GT and Generated SQL were executable and SQL was expected
                if test_case["ground_truth_sql"] is not None and eval_results["ground_truth_execution_success"] and eval_results["sql_execution_success"]:
                    match, diff_detail = compare_query_results(ground_truth_sql_results_rows, generated_sql_results_rows)
                    eval_results["sql_execution_match"] = match
                    if not match: eval_results["errors"].append(f"SQL result mismatch: {diff_detail}")
                elif test_case["ground_truth_sql"] is not None: # If SQL was expected but one failed
                     if eval_results["ground_truth_execution_success"]:
                         eval_results["errors"].append("Generated SQL failed to execute, cannot match results.")
                     elif eval_results["sql_execution_success"]:
                         eval_results["errors"].append("Ground Truth SQL failed to execute, cannot match results.")
                     else:
                         eval_results["errors"].append("Both GT and Generated SQL failed to execute.")
                # If test_case["ground_truth_sql"] is None, sql_execution_match should implicitly be True or N/A
                else: # This block is for cases where no SQL is expected (e.g., greetings, farewells)
                    eval_results["sql_execution_match"] = True
                    eval_results["sql_execution_success"] = (eval_results["generated_sql"] is None) # Success if generated SQL is also None
                    eval_results["ground_truth_execution_success"] = True # Always true if no GT SQL expected
                    if eval_results["generated_sql"] is not None:
                        eval_results["errors"].append("Generated SQL when none was expected.")
                        eval_results["sql_execution_success"] = False # Indicate failure if SQL was generated unexpectedly

            except Exception as e:
                eval_results["errors"].append(f"Error during SQL execution phase: {e}")
                logger.critical(f"Test Case {test_case['id']}: SQL execution phase error: {e}", exc_info=True)
            finally:
                if conn: conn.close()
        else: # Case where ground_truth_sql is None (e.g., greetings)
            eval_results["sql_execution_match"] = True
            eval_results["sql_execution_success"] = (eval_results["generated_sql"] is None) # Success if generated SQL is also None
            eval_results["ground_truth_execution_success"] = True # Always true if no GT SQL expected
            if eval_results["generated_sql"] is not None:
                eval_results["errors"].append("Generated SQL when none was expected.")
                eval_results["sql_execution_success"] = False # Indicate failure if SQL was generated unexpectedly

        # 3. Get NL Response (re-call the bot with results if SQL was executed)
        # This part ensures that if a SQL query was actually run and returned results,
        # the bot is re-prompted to generate NL *based on those results*.
        # For non-SQL queries, the NL might have already been captured in the first call.
        if eval_results["generated_sql"] and eval_results["sql_execution_success"] and generated_sql_results_with_headers:
            # If a SQL query was successfully generated and executed, re-call the bot for the NL based on results
            final_response_from_bot_nl_phase = get_gemini_response_func(
                test_case["question"], model_obj, conversation_history_list=[], # Use fresh history for NL generation based on specific results
                query_results_with_headers=generated_sql_results_with_headers
            )
            if final_response_from_bot_nl_phase and "natural_language_response" in final_response_from_bot_nl_phase:
                eval_results["generated_nl_response"] = final_response_from_bot_nl_phase["natural_language_response"]
                logger.debug(f"Test Case {test_case['id']}: NL response regenerated with query results.")
            else:
                eval_results["errors"].append(f"Bot failed to generate NL response from results: {final_response_from_bot_nl_phase.get('error', 'Unknown error')}")
                logger.error(f"Test Case {test_case['id']}: Bot failed NL generation from results.")
        # else: NL response might have been captured in the first call or remains None if no SQL was relevant.

        # 4. Get LLM Judge's Verdict
        # Always run the LLM judge if we have enough information, even if there were errors,
        # so the judge can assess the quality of the error responses.
        llm_judge_prompt = get_llm_judge_prompt(
            test_case["question"], eval_results["generated_sql"], test_case["ground_truth_sql"],
            eval_results["generated_nl_response"] or "N/A (Bot failed to generate NL)",
            test_case.get("expected_nl_response", "Not Provided"),
            generated_sql_results_with_headers or {}, # Pass empty dict if no results
            get_schema_string_func()
        )
        llm_judge_response = get_llm_judge_response(model_obj, llm_judge_prompt, logger)
        eval_results["llm_judge_verdict"] = llm_judge_response
        if "error" in llm_judge_response:
            eval_results["errors"].append(f"LLM Judge error: {llm_judge_response['error']}")
            logger.error(f"Test Case {test_case['id']}: LLM Judge error.")
        else:
            logger.info(f"Test Case {test_case['id']}: LLM Judge verdict received.")

    except Exception as e:
        eval_results["errors"].append(f"An unexpected error during test case evaluation: {e}")
        logger.critical(f"Test Case {test_case['id']}: Unexpected error: {e}", exc_info=True)

    return eval_results

# --- Main Evaluation Runner Function ---
def run_evaluation(model_obj, get_gemini_response_func, get_db_connection_func, get_schema_string_func, logger, evaluation_output_dir, antipolo_tz):
    """Runs the full evaluation suite against the TEST_CASES."""
    eval_logger.info("Starting evaluation process...")
    
    total_cases = len(TEST_CASES)
    overall_results = {
        "summary": {
            "timestamp": datetime.now(antipolo_tz).strftime('%Y-%m-%d %H:%M:%S %Z%z'),
            "total_test_cases": total_cases, 
            "sql_execution_match_count": 0, # Match on essential columns
            "nl_response_present_count": 0,
            "llm_judge_success_count": 0,
            "cases_where_sql_expected": 0,
            "cases_where_sql_not_expected": 0,
            "sql_generation_expected_succeeded": 0, # SQL was expected and generated successfully
            "sql_generation_unexpected_generated": 0, # SQL was NOT expected but was generated
            "sql_generation_expected_failed": 0, # SQL was expected but generation failed
            "ground_truth_execution_success_count": 0,
            "generated_sql_execution_success_count": 0,
            "successful_end_to_end_cases": 0, # This should be the true measure of user success
            
            # NEW COMPREHENSIVENESS METRICS
            "sql_comprehensiveness_verdict_counts": {
                "Exact Match": 0,
                "Beneficial Superset": 0,
                "Irrelevant Superset": 0,
                "Missing Essential Columns": 0,
                "Incorrect Filtering/Logic": 0,
                "No SQL Expected/Generated Correctly": 0,
                "N/A (Execution Failed)": 0,
                "Generated SQL Unexpected": 0
            },
            # NEW NL SCORE AVERAGES
            "avg_nl_accuracy_score": 0.0,
            "avg_nl_helpfulness_score": 0.0,
            "nl_score_calculation_count": 0 # To get accurate averages
        },
        "details": []
    }

    for i, test_case in enumerate(TEST_CASES):
        eval_logger.info(f"Processing Test Case {i+1}/{total_cases}: ID={test_case['id']} - '{test_case['question']}'")
        
        # Pass a callable to get_ground_truth_headers for DB connection, or pass the connection func directly
        result = evaluate_test_case(test_case, model_obj, get_gemini_response_func, get_db_connection_func, get_schema_string_func, logger)
        overall_results["details"].append(result)

        # Update summary metrics
        if test_case["ground_truth_sql"] is not None:
            overall_results["summary"]["cases_where_sql_expected"] += 1
            if result["generated_sql"] is not None:
                overall_results["summary"]["sql_generation_expected_succeeded"] += 1
            else:
                overall_results["summary"]["sql_generation_expected_failed"] += 1
        else:
            overall_results["summary"]["cases_where_sql_not_expected"] += 1
            if result["generated_sql"] is not None:
                overall_results["summary"]["sql_generation_unexpected_generated"] += 1

        if result["ground_truth_execution_success"]:
            overall_results["summary"]["ground_truth_execution_success_count"] += 1
        
        if result["sql_execution_success"]:
            overall_results["summary"]["generated_sql_execution_success_count"] += 1
            
        if result["sql_execution_match"]: # This now reflects essential column matching only
            overall_results["summary"]["sql_execution_match_count"] += 1

        if result["generated_nl_response"]:
            overall_results["summary"]["nl_response_present_count"] += 1

        if result["llm_judge_verdict"] and "error" not in result["llm_judge_verdict"]:
            overall_results["summary"]["llm_judge_success_count"] += 1
            
            # Update Comprehensiveness Verdict Counts
            comprehensiveness_verdict = result["llm_judge_verdict"].get("sql_comprehensiveness_verdict")
            if comprehensiveness_verdict in overall_results["summary"]["sql_comprehensiveness_verdict_counts"]:
                overall_results["summary"]["sql_comprehensiveness_verdict_counts"][comprehensiveness_verdict] += 1
            else:
                overall_results["summary"]["sql_comprehensiveness_verdict_counts"][comprehensiveness_verdict] = 1 # Catch-all for new verdicts

            # Update NL Score Averages
            nl_acc_score = result["llm_judge_verdict"].get("nl_accuracy_score")
            nl_help_score = result["llm_judge_verdict"].get("nl_helpfulness_score")
            if nl_acc_score is not None and nl_help_score is not None:
                overall_results["summary"]["avg_nl_accuracy_score"] += nl_acc_score
                overall_results["summary"]["avg_nl_helpfulness_score"] += nl_help_score
                overall_results["summary"]["nl_score_calculation_count"] += 1

        # Define "successful_end_to_end_cases" more precisely with user-centric view
        is_successful_e2e = True
        
        # 1. SQL correctness and generation success (for cases where SQL is expected)
        if test_case["ground_truth_sql"] is not None: 
            # If SQL was expected, we need generated SQL, it must execute, and LLM judge must deem it functional/beneficial
            if not result["generated_sql"] or \
               not result["sql_execution_success"] or \
               (result["llm_judge_verdict"] and result["llm_judge_verdict"].get("sql_comprehensiveness_verdict") in ["Missing Essential Columns", "Incorrect Filtering/Logic"]):
                is_successful_e2e = False
        else: # SQL was NOT expected
            if result["generated_sql"] is not None: # Failed if SQL was unexpectedly generated
                is_successful_e2e = False
            
        # 2. NL Response must be present and highly accurate/helpful by LLM judge
        if not result["generated_nl_response"] or \
           not (result["llm_judge_verdict"] and "error" not in result["llm_judge_verdict"] and \
                result["llm_judge_verdict"].get("nl_accuracy_score", 0) >= 4 and \
                result["llm_judge_verdict"].get("nl_helpfulness_score", 0) >= 4): # Set your threshold for "passing" NL scores (e.g., 4 or 5)
            is_successful_e2e = False
        
        # 3. LLM Judge itself must not have errored out
        if result["llm_judge_verdict"] and "error" in result["llm_judge_verdict"]:
            is_successful_e2e = False

        if is_successful_e2e:
            overall_results["summary"]["successful_end_to_end_cases"] += 1
            eval_logger.info(f"Test Case {test_case['id']}: Marked as END-TO-END SUCCESS.")
        else:
            eval_logger.warning(f"Test Case {test_case['id']}: NOT an END-TO-END SUCCESS. Errors: {result['errors']}")
            if result["llm_judge_verdict"] and "error" not in result["llm_judge_verdict"]:
                eval_logger.warning(f"LLM Judge Scores for {test_case['id']}: SQL={result['llm_judge_verdict'].get('sql_correctness_score')}, Comp_Verdict={result['llm_judge_verdict'].get('sql_comprehensiveness_verdict')}, NL_Acc={result['llm_judge_verdict'].get('nl_accuracy_score')}, NL_Help={result['llm_judge_verdict'].get('nl_helpfulness_score')}")
        
        if result["errors"]: eval_logger.warning(f"Detailed errors for Test Case {test_case['id']}: {result['errors']}")

    # Calculate final averages
    if overall_results["summary"]["nl_score_calculation_count"] > 0:
        overall_results["summary"]["avg_nl_accuracy_score"] /= overall_results["summary"]["nl_score_calculation_count"]
        overall_results["summary"]["avg_nl_helpfulness_score"] /= overall_results["summary"]["nl_score_calculation_count"]

    eval_logger.info("Evaluation complete.")
    eval_logger.info(f"Overall Evaluation Results Summary:\n{json.dumps(overall_results['summary'], indent=2)}")
    timestamp = datetime.now(antipolo_tz).strftime('%Y%m%d_%H%M%S')
    
    json_filename = os.path.join(evaluation_output_dir, f"evaluation_results_{timestamp}.json")
    try:
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(overall_results, f, ensure_ascii=False, indent=4)
        eval_logger.info(f"Evaluation results saved to {json_filename}")
    except Exception as e:
        eval_logger.error(f"Failed to save JSON evaluation results: {e}", exc_info=True)

    csv_filename = os.path.join(evaluation_output_dir, f"evaluation_details_{timestamp}.csv")
    try:
        csv_headers = [
            "id", "question", "language", "query_type", "generated_sql", "ground_truth_sql",
            "sql_execution_success", "ground_truth_execution_success", "sql_execution_match",
            "generated_nl_response", "expected_nl_response",
            "llm_judge_sql_correctness_score", "llm_judge_sql_comprehensiveness_verdict", "llm_judge_sql_explanation", # CHANGED
            "llm_judge_nl_accuracy_score", "llm_judge_nl_helpfulness_score", "llm_judge_nl_explanation",
            "llm_judge_overall_comment", "errors"
        ]
        
        csv_rows = []
        for detail in overall_results["details"]:
            row = {k: detail.get(k) for k in ["id", "question", "language", "query_type", "generated_sql", "ground_truth_sql",
                                              "sql_execution_success", "ground_truth_execution_success", "sql_execution_match",
                                              "generated_nl_response", "expected_nl_response"]}
            row["errors"] = "; ".join(detail.get("errors", []))
            
            judge_verdict = detail.get("llm_judge_verdict", {})
            row["llm_judge_sql_correctness_score"] = judge_verdict.get("sql_correctness_score")
            row["llm_judge_sql_comprehensiveness_verdict"] = judge_verdict.get("sql_comprehensiveness_verdict") # CHANGED
            row["llm_judge_sql_explanation"] = judge_verdict.get("sql_explanation")
            row["llm_judge_nl_accuracy_score"] = judge_verdict.get("nl_accuracy_score")
            row["llm_judge_nl_helpfulness_score"] = judge_verdict.get("nl_helpfulness_score")
            row["llm_judge_nl_explanation"] = judge_verdict.get("nl_explanation")
            row["llm_judge_overall_comment"] = judge_verdict.get("overall_comment")
            
            csv_rows.append(row)

        with open(csv_filename, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(csv_rows)
        eval_logger.info(f"Detailed evaluation results saved to {csv_filename}")
    except Exception as e:
        eval_logger.error(f"Failed to save CSV evaluation results: {e}", exc_info=True)