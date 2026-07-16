import json
import logging
import time
from pathlib import Path
import re

from src.rag.generate import answer_question

logger = logging.getLogger(__name__)

EVAL_FILE = Path("data/eval_questions.json")
RESULTS_FILE = Path("data/eval_results.json")

def load_questions():
    """
    Load evaluation questions.
    """

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
    
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    questions = load_questions()

    print(f"Loaded {len(questions)} evaluation questions.")



def keyword_score(
    answer: str,
    expected_keywords: list[str],
) -> float:
    """
    Compute the fraction of expected keywords found
    in the generated answer.

    Numeric answers are evaluated primarily using the
    first keyword (expected numeric value).

    Qualitative answers use all expected keywords.
    """

    if not expected_keywords:
        return 0.0

    def normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[$,%]", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    normalized_answer = normalize(answer)

    normalized_keywords = [
        normalize(keyword)
        for keyword in expected_keywords
    ]

    # ----------------------------
    # Numeric Question
    # ----------------------------

    if any(char.isdigit() for char in normalized_keywords[0]):

        return (
            1.0
            if normalized_keywords[0] in normalized_answer
            else 0.0
        )

    # ----------------------------
    # Qualitative Question
    # ----------------------------

    matches = sum(
        keyword in normalized_answer
        for keyword in normalized_keywords
    )

    return round(
        matches / len(normalized_keywords),
        2,
    )

    def normalize(text: str) -> str:
        text = text.lower()

        text = re.sub(r"[$,%]", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    normalized_answer = normalize(answer)

    matches = 0

    for keyword in expected_keywords:

        normalized_keyword = normalize(keyword)

        if normalized_keyword in normalized_answer:
            matches += 1

    return round(
        matches / len(expected_keywords),
        2,
    )

def has_citation(result: dict) -> bool:
    """
    Return True if at least one source
    citation exists.
    """

    return len(result["sources"]) > 0

def measure_latency(
    question: str,
    ticker: str,
):
    """
    Run one question while measuring
    response time.
    """

    start = time.perf_counter()

    result = answer_question(
        question=question,
        ticker_filter=ticker,
    )

    latency = round(
        time.perf_counter() - start,
        2,
    )

    return result, latency

def pass_fail(
    keyword_score: float,
    confidence: float,
    has_source: bool,
    difficulty: str,
) -> bool:
    """
    Determine whether an evaluation passes.

    Thresholds vary based on question difficulty.
    """

    thresholds = {
        "easy": {
            "keyword": 0.60,
            "confidence": 0.70,
        },
        "medium": {
            "keyword": 0.50,
            "confidence": 0.60,
        },
        "hard": {
            "keyword": 0.40,
            "confidence": 0.50,
        },
    }

    level = thresholds[difficulty]

    return (
        keyword_score >= level["keyword"]
        and confidence >= level["confidence"]
        and has_source
    )

def run_eval() -> list[dict]:
    """
    Run the complete evaluation benchmark.

    Returns:
        List containing evaluation results for every question.
    """

    questions = load_questions()

    logger.info(
        "Loaded %d evaluation questions.",
        len(questions),
    )

    results = []

    for index, question in enumerate(
        questions,
        start=1,
    ):

        logger.info(
            "[%d/%d] %s",
            index,
            len(questions),
            question["question"],
        )

        try:

            result, latency = measure_latency(
                question=question["question"],
                ticker=question["ticker"],
            )

            score = keyword_score(
                answer=result["answer"],
                expected_keywords=question["expected_answer_contains"],
            )

            citation = has_citation(result)

            passed = pass_fail(
                keyword_score=score,
                confidence=result["confidence"],
                has_source=citation,
                difficulty=question["difficulty"],
            )

            # ----------------------------------------
            # Failure Reason
            # ----------------------------------------

            failure_reason = None

            if not passed:

                keyword_threshold = {
                    "easy": 0.60,
                    "medium": 0.50,
                    "hard": 0.40,
                }

                confidence_threshold = {
                    "easy": 0.70,
                    "medium": 0.60,
                    "hard": 0.50,
                }

                if not citation:
                    failure_reason = "Missing citation"

                elif score < keyword_threshold[
                    question["difficulty"]
                ]:
                    failure_reason = "Low keyword score"

                elif result["confidence"] < confidence_threshold[
                    question["difficulty"]
                ]:
                    failure_reason = "Low confidence"

                else:
                    failure_reason = "Unknown"

            results.append(
                {
                    "id": question["id"],
                    "ticker": question["ticker"],
                    "year": question["year"],
                    "category": question["category"],
                    "difficulty": question["difficulty"],
                    "answer_type": question["answer_type"],
                    "description": question["description"],
                    "question": question["question"],
                    "ground_truth": question["ground_truth"],
                    "ground_truth_snippet": question[
                        "ground_truth_snippet"
                    ],
                    "answer": result["answer"],
                    "confidence": result["confidence"],
                    "keyword_score": score,
                    "latency": latency,
                    "has_source": citation,
                    "num_sources": len(result["sources"]),
                    "sources": result["sources"],
                    "passed": passed,
                    "failure_reason": failure_reason,
                }
            )

            logger.info(
                "✓ %.2f | Conf: %.2f | %.2fs | %s",
                score,
                result["confidence"],
                latency,
                "PASS" if passed else "FAIL",
            )

        except Exception as error:

            logger.exception(
                "Evaluation failed for question %s",
                question["id"],
            )

            results.append(
                {
                    "id": question["id"],
                    "ticker": question["ticker"],
                    "year": question["year"],
                    "category": question["category"],
                    "difficulty": question["difficulty"],
                    "answer_type": question["answer_type"],
                    "description": question["description"],
                    "question": question["question"],
                    "ground_truth": question["ground_truth"],
                    "ground_truth_snippet": question[
                        "ground_truth_snippet"
                    ],
                    "answer": "",
                    "confidence": 0.0,
                    "keyword_score": 0.0,
                    "latency": 0.0,
                    "has_source": False,
                    "num_sources": 0,
                    "sources": [],
                    "passed": False,
                    "failure_reason": str(error),
                }
            )

    return results

def compute_statistics(results: list[dict]) -> dict:
    """
    Compute summary statistics for all evaluation results.

    Args:
        results:
            List returned by run_eval().

    Returns:
        Dictionary containing overall metrics and breakdowns.
    """

    if not results:
        return {
            "total_questions": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "average_keyword_score": 0.0,
            "average_confidence": 0.0,
            "average_latency": 0.0,
            "citation_rate": 0.0,
            "company_breakdown": {},
            "difficulty_breakdown": {},
            "category_breakdown": {},
        }

    total_questions = len(results)

    passed = 0
    total_keyword_score = 0.0
    total_confidence = 0.0
    total_latency = 0.0
    total_citations = 0

    company_breakdown = {}
    difficulty_breakdown = {}
    category_breakdown = {}

    for result in results:

        # ---------- Overall metrics ----------

        if result["passed"]:
            passed += 1

        total_keyword_score += result["keyword_score"]
        total_confidence += result["confidence"]
        total_latency += result["latency"]

        if result["has_source"]:
            total_citations += 1

        # ---------- Company ----------

        ticker = result["ticker"]

        if ticker not in company_breakdown:
            company_breakdown[ticker] = {
                "total": 0,
                "passed": 0,
            }

        company_breakdown[ticker]["total"] += 1

        if result["passed"]:
            company_breakdown[ticker]["passed"] += 1

        # ---------- Difficulty ----------

        difficulty = result["difficulty"]

        if difficulty not in difficulty_breakdown:
            difficulty_breakdown[difficulty] = {
                "total": 0,
                "passed": 0,
            }

        difficulty_breakdown[difficulty]["total"] += 1

        if result["passed"]:
            difficulty_breakdown[difficulty]["passed"] += 1

        # ---------- Category ----------

        category = result["category"]

        if category not in category_breakdown:
            category_breakdown[category] = {
                "total": 0,
                "passed": 0,
            }

        category_breakdown[category]["total"] += 1

        if result["passed"]:
            category_breakdown[category]["passed"] += 1

    failed = total_questions - passed

    statistics = {
        "total_questions": total_questions,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total_questions, 2),
        "average_keyword_score": round(
            total_keyword_score / total_questions,
            2,
        ),
        "average_confidence": round(
            total_confidence / total_questions,
            2,
        ),
        "average_latency": round(
            total_latency / total_questions,
            2,
        ),
        "citation_rate": round(
            total_citations / total_questions,
            2,
        ),
        "company_breakdown": company_breakdown,
        "difficulty_breakdown": difficulty_breakdown,
        "category_breakdown": category_breakdown,
    }

    return statistics

def save_results(
    results: list[dict],
    statistics: dict,
):
    """
    Save evaluation results to disk.
    """

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "results": results,
        "statistics": statistics,
    }

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
            ensure_ascii=False,
        )

    logger.info(
        "Saved evaluation results to %s",
        RESULTS_FILE,
    )

def print_report(
    statistics: dict,
    results: list[dict],
    ):
    """
    Print a formatted evaluation report.
    """

    print("\n" + "=" * 80)
    print("FinSight Evaluation Report")
    print("=" * 80)

    print(f"Questions Evaluated : {statistics['total_questions']}")
    print(f"Passed              : {statistics['passed']}")
    print(f"Failed              : {statistics['failed']}")
    print(f"Pass Rate           : {statistics['pass_rate']:.2%}")

    print()

    print(f"Average Keyword     : {statistics['average_keyword_score']:.2f}")
    print(f"Average Confidence  : {statistics['average_confidence']:.2f}")
    print(f"Average Latency     : {statistics['average_latency']:.2f} sec")
    print(f"Citation Rate       : {statistics['citation_rate']:.2%}")

    # --------------------------------------------------
    # Company Breakdown
    # --------------------------------------------------

    print("\n" + "-" * 80)
    print("Company Breakdown")
    print("-" * 80)

    for company, values in statistics["company_breakdown"].items():

        print(
            f"{company:<8}"
            f"{values['passed']}/{values['total']} passed"
        )

    # --------------------------------------------------
    # Difficulty Breakdown
    # --------------------------------------------------

    print("\n" + "-" * 80)
    print("Difficulty Breakdown")
    print("-" * 80)

    for difficulty, values in statistics["difficulty_breakdown"].items():

        print(
            f"{difficulty:<10}"
            f"{values['passed']}/{values['total']} passed"
        )

    # --------------------------------------------------
    # Category Breakdown
    # --------------------------------------------------

    print("\n" + "-" * 80)
    print("Category Breakdown")
    print("-" * 80)

    for category, values in statistics["category_breakdown"].items():

        print(
            f"{category:<12}"
            f"{values['passed']}/{values['total']} passed"
        )

    # --------------------------------------------------
    # Failed Questions
    # --------------------------------------------------

    failed_results = [
        result
        for result in results
        if not result["passed"]
    ]

    print("\n" + "-" * 80)
    print("Failed Questions")
    print("-" * 80)

    if not failed_results:

        print("None 🎉")

    else:

        for result in failed_results:

            print(
                f"\n{result['id']}"
            )

            print(
                f"Question       : {result['question']}"
            )

            print(
                f"Failure Reason : {result['failure_reason']}"
            )

            print(
                f"Keyword Score  : {result['keyword_score']:.2f}"
            )

            print(
                f"Confidence     : {result['confidence']:.2f}"
            )

    print("\n" + "=" * 80)

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    logger.info("Starting evaluation...")

    results = run_eval()

    statistics = compute_statistics(results)

    save_results(
        results,
        statistics,
    )

    print_report(statistics,
                 results,)

    logger.info("Evaluation completed successfully.")