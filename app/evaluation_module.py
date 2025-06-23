import os
import json
import sqlite3
import csv
from datetime import datetime
import logging

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

eval_logger.info("Evaluation module loaded.")


# --- Test Dataset for Evaluation ---
TEST_CASES = [
    {
        "id": "1",
        "question": "What were my top 3 largest withdrawals?",
        "language": "en",
        "query_type": "top_transactions",
        # Kept minimal columns for GT as per previous discussion, expecting LLM Judge to handle extra columns gracefully
        "ground_truth_sql": "SELECT Date_Timestamp, \"Transaction_Details\", Withdrawal_Amount FROM bank_transactions WHERE Withdrawal_Amount > 0 ORDER BY Withdrawal_Amount DESC LIMIT 3;",
        "expected_nl_response": "Your top 3 largest withdrawals were [list details]. Let me know if you need more details!"
    },
    {
        "id": "2",
        "question": "How much did I deposit in total for 2024?",
        "language": "en",
        "query_type": "sum_deposits",
        "ground_truth_sql": "SELECT SUM(COALESCE(Deposit_Amount, 0)) FROM bank_transactions WHERE strftime('%Y', Date_Timestamp) = '2024';",
        "expected_nl_response": "In 2024, you deposited a total of ₱[amount]."
    },
    {
        "id": "3",
        "question": "List all ATM withdrawals in January 2025.",
        "language": "en",
        "query_type": "filtered_transactions",
        "ground_truth_sql": "SELECT Date_Timestamp, \"Merchant_Counterparty_Name\", Withdrawal_Amount FROM bank_transactions WHERE \"Transaction_Details\" LIKE '%ATM WITHDRAWAL%' AND strftime('%Y-%m', Date_Timestamp) = '2025-01' ORDER BY Date_Timestamp ASC;",
        "expected_nl_response": "Here are all your ATM withdrawals for January 2025: [list details]."
    }
]

# --- Helper for Comparing Query Results (for Execution Accuracy) ---
def compare_query_results(result1_rows, result2_rows, column_order_independent=True):
    """Compares two sets of query results for semantic equivalence."""
    if len(result1_rows) != len(result2_rows):
        return False, f"Row count mismatch: {len(result1_rows)} vs {len(result2_rows)}"

    canonical_result1 = []
    for row in result1_rows:
        canonical_result1.append(frozenset(str(col) for col in row) if column_order_independent else tuple(str(col) for col in row))
    
    canonical_result2 = []
    for row in result2_rows:
        canonical_result2.append(frozenset(str(col) for col in row) if column_order_independent else tuple(str(col) for col in row))

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
        table_lines = [f"| {' | '.join(headers)} |", f"|{'-' * (sum(len(h) for h in headers) + 3 * len(headers) - 1)}|"]
        table_lines.extend(f"| {' | '.join(str(col) for col in row)} |" for row in rows)
        results_str = "\n".join(table_lines)

    return f"""
You are an expert evaluator for a banking chatbot. Critically assess the generated SQL query and natural language response based on the user's question and schema.

**Instructions:**
1.  **Evaluate Generated SQL vs. Ground Truth SQL:**
    * **Semantic Equivalence:** Does `Generated SQL` produce the exact same result set as `Ground Truth SQL` given `User Question` and `Database Schema`?
    * **SQL Correctness Score:** Rate 1 (completely wrong) to 5 (perfectly equivalent).
2.  **Evaluate Natural Language Response:**
    * **Accuracy & Helpfulness:** Is `Generated Natural Language Response` accurate, helpful, conversational, and correctly answers `User Question` based on `SQL Query Results`? Compare to `Expected Natural Language Response`.
    * **NL Accuracy Score:** Rate 1 (completely inaccurate) to 5 (fully accurate).
    * **NL Helpfulness Score:** Rate 1 (not helpful) to 5 (very helpful and conversational).
3.  **Provide concise explanations for all scores.**
4.  **Output JSON format.**

**Input Data:**
User Question: {user_question}
Database Schema:\n{schema_str}
Ground Truth SQL: {ground_truth_sql}
Generated SQL: {generated_sql}
SQL Query Results (from Generated SQL):\n{results_str}
Generated Natural Language Response: {generated_nl_response}
Expected Natural Language Response (for reference): {expected_nl_response if expected_nl_response else "Not provided"}

**Output Format (JSON):**
{{
  "sql_correctness_score": [1-5],
  "sql_equivalence_verdict": "Yes" | "No" | "Cannot Determine",
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
        content = model_obj.generate_content(prompt_text).text.strip()
        logger.debug(f"Raw LLM Judge response: {content[:500]}...")

        if content.startswith("```json"):
            json_str = content.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        elif content.startswith("```"):
            json_str = content.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        else:
            json_str = content.strip()

        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"LLM Judge JSON decoding error: {e}. Raw content: {content}", exc_info=True)
        return {"error": f"JSON decoding error: {e}", "raw_content": content}
    except Exception as e:
        logger.error(f"LLM Judge unexpected error: {e}", exc_info=True)
        return {"error": f"Unexpected error: {e}"}


# --- Evaluation Function for a Single Test Case ---
def evaluate_test_case(test_case: dict, model_obj, get_gemini_response_func, get_db_connection_func, get_schema_string_func, logger):
    """Evaluates a single test case."""
    eval_results = {
        "id": test_case["id"], "question": test_case["question"], "language": test_case["language"],
        "query_type": test_case.get("query_type", "unknown"), "generated_sql": None,
        "ground_truth_sql": test_case["ground_truth_sql"], "sql_execution_success": False,
        "ground_truth_execution_success": False, "sql_execution_match": False,
        "generated_nl_response": None, "expected_nl_response": test_case.get("expected_nl_response"),
        "llm_judge_verdict": None, "errors": []
    }

    generated_sql_results_with_headers, ground_truth_sql_results_rows = None, []
    conn = None

    try:
        # 1. Get Generated SQL from the main bot
        response_from_bot = get_gemini_response_func(test_case["question"], model_obj, test_case["language"])
        if response_from_bot and "sql" in response_from_bot:
            eval_results["generated_sql"] = response_from_bot["sql"]
        else:
            eval_results["errors"].append(f"Bot failed to generate SQL: {response_from_bot.get('error', 'Unknown error')}")
            logger.error(f"Test Case {test_case['id']}: Bot failed SQL generation.")
            return eval_results

        # 2. Execute Ground Truth and Generated SQL queries
        try:
            conn = get_db_connection_func()
            cursor = conn.cursor()

            try:
                cursor.execute(test_case["ground_truth_sql"])
                ground_truth_sql_results_rows = cursor.fetchall()
                eval_results["ground_truth_execution_success"] = True
                logger.debug(f"Test Case {test_case['id']}: GT SQL executed.")
            except sqlite3.Error as e:
                eval_results["errors"].append(f"GT SQL execution error: {e}")
                eval_results["ground_truth_execution_success"] = False
                logger.error(f"Test Case {test_case['id']}: GT SQL error: {e}")

            if eval_results["generated_sql"]:
                try:
                    cursor.execute(eval_results["generated_sql"])
                    generated_sql_results_rows = cursor.fetchall()
                    eval_results["sql_execution_success"] = True
                    column_headers = [desc[0] for desc in cursor.description]
                    generated_sql_results_with_headers = {"headers": column_headers, "rows": [list(row) for row in generated_sql_results_rows]}
                    logger.debug(f"Test Case {test_case['id']}: Generated SQL executed.")
                except sqlite3.Error as e:
                    eval_results["errors"].append(f"Generated SQL execution error: {e}")
                    eval_results["sql_execution_success"] = False
                    logger.error(f"Test Case {test_case['id']}: Generated SQL error: {e}")
            else:
                eval_results["errors"].append("Generated SQL missing, skipping execution.")

            if eval_results["ground_truth_execution_success"] and eval_results["sql_execution_success"]:
                match, diff_detail = compare_query_results(ground_truth_sql_results_rows, generated_sql_results_rows)
                eval_results["sql_execution_match"] = match
                if not match: eval_results["errors"].append(f"SQL result mismatch: {diff_detail}")
            elif eval_results["ground_truth_execution_success"]:
                 eval_results["errors"].append("Generated SQL failed to execute, cannot match results.")
            elif eval_results["sql_execution_success"]:
                 eval_results["errors"].append("Ground Truth SQL failed to execute, cannot match results.")

        except Exception as e:
            eval_results["errors"].append(f"Error during SQL execution phase: {e}")
            logger.critical(f"Test Case {test_case['id']}: SQL execution phase error: {e}", exc_info=True)
        finally:
            if conn: conn.close()

        # 3. Get NL Response
        if eval_results["sql_execution_success"] and generated_sql_results_with_headers:
            final_response_from_bot = get_gemini_response_func(
                test_case["question"], model_obj, test_case["language"], query_results_with_headers=generated_sql_results_with_headers
            )
            if final_response_from_bot and "natural_language_response" in final_response_from_bot:
                eval_results["generated_nl_response"] = final_response_from_bot["natural_language_response"]
                logger.debug(f"Test Case {test_case['id']}: NL response generated.")
            else:
                eval_results["errors"].append(f"Bot failed to generate NL response: {final_response_from_bot.get('error', 'Unknown error')}")
                logger.error(f"Test Case {test_case['id']}: Bot failed NL generation.")
        else:
            eval_results["errors"].append("Skipping NL generation due to SQL issues.")

        # 4. Get LLM Judge's Verdict
        if eval_results["generated_sql"]:
            llm_judge_prompt = get_llm_judge_prompt(
                test_case["question"], eval_results["generated_sql"], test_case["ground_truth_sql"],
                eval_results["generated_nl_response"] or "N/A (Bot failed to generate NL)",
                test_case.get("expected_nl_response", "Not Provided"),
                generated_sql_results_with_headers or {}, get_schema_string_func()
            )
            llm_judge_response = get_llm_judge_response(model_obj, llm_judge_prompt, logger)
            eval_results["llm_judge_verdict"] = llm_judge_response
            if "error" in llm_judge_response:
                eval_results["errors"].append(f"LLM Judge error: {llm_judge_response['error']}")
                logger.error(f"Test Case {test_case['id']}: LLM Judge error.")
            else:
                logger.info(f"Test Case {test_case['id']}: LLM Judge verdict received.")
        else:
            eval_results["errors"].append("Skipping LLM Judge due to missing generated SQL.")

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
            "total_test_cases": total_cases, "sql_execution_match_count": 0,
            "failed_sql_generation": 0, "failed_sql_execution": 0,
            "failed_ground_truth_execution": 0, "failed_nl_generation": 0,
            "failed_llm_judge_response": 0, "successful_end_to_end_cases": 0
        },
        "details": []
    }

    for i, test_case in enumerate(TEST_CASES):
        eval_logger.info(f"Processing Test Case {i+1}/{total_cases}: ID={test_case['id']} - '{test_case['question']}'")
        result = evaluate_test_case(test_case, model_obj, get_gemini_response_func, get_db_connection_func, get_schema_string_func, logger)
        overall_results["details"].append(result)

        if result["generated_sql"] is None: overall_results["summary"]["failed_sql_generation"] += 1
        if not result["ground_truth_execution_success"]: overall_results["summary"]["failed_ground_truth_execution"] += 1
        if not result["sql_execution_success"]: overall_results["summary"]["failed_sql_execution"] += 1
        elif result["sql_execution_match"]: overall_results["summary"]["sql_execution_match_count"] += 1
        if result["generated_nl_response"] is None and result["sql_execution_success"]: overall_results["summary"]["failed_nl_generation"] += 1
        if result["llm_judge_verdict"] and "error" in result["llm_judge_verdict"]: overall_results["summary"]["failed_llm_judge_response"] += 1

        is_successful_e2e = (
            result["generated_sql"] and result["sql_execution_success"] and result["sql_execution_match"] and
            result["generated_nl_response"] and result["ground_truth_execution_success"] and
            not (result["llm_judge_verdict"] and "error" in result["llm_judge_verdict"])
        )
        if is_successful_e2e: overall_results["summary"]["successful_end_to_end_cases"] += 1
            
        if result["errors"]: eval_logger.warning(f"Errors in Test Case {test_case['id']}: {result['errors']}")

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
            "llm_judge_sql_correctness_score", "llm_judge_sql_equivalence_verdict", "llm_judge_sql_explanation",
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
            for judge_key in ["sql_correctness_score", "sql_equivalence_verdict", "sql_explanation",
                              "nl_accuracy_score", "nl_helpfulness_score", "nl_explanation", "overall_comment"]:
                row[f"llm_judge_{judge_key}"] = judge_verdict.get(judge_key)
            
            csv_rows.append(row)

        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(csv_rows)
        eval_logger.info(f"Detailed evaluation results saved to {csv_filename}")
    except Exception as e:
        eval_logger.error(f"Failed to save CSV evaluation results: {e}", exc_info=True)

    return {
        "message": "Evaluation complete. Results are available in logs/evaluation.log and saved to files.",
        "json_file": os.path.abspath(json_filename),
        "csv_file": os.path.abspath(csv_filename),
        "summary": overall_results["summary"]
    }