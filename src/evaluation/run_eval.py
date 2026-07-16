import json

from src.rag.generate import answer_question


def run_evaluation():

    with open(
        "data/eval_questions.json",
        "r",
        encoding="utf-8",
    ) as f:

        tests = json.load(f)

    print("\n" + "=" * 100)
    print("FinSight Evaluation")
    print("=" * 100)

    for i, test in enumerate(tests, start=1):

        print(f"\nTest {i}")

        print("-" * 100)

        print("Ticker    :", test["ticker"])
        print("Question  :", test["question"])

        result = answer_question(
            question=test["question"],
            ticker_filter=test["ticker"],
        )

        print("\nConfidence :", result["confidence"])

        print("\nAnswer")
        print(result["answer"])

        print("\nCitations")
        print(result["citations"])

        print("\n" + "=" * 100)


if __name__ == "__main__":
    run_evaluation()