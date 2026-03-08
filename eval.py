import json
from main import run_query


def run_evaluation():

    test_queries = [
        {
            "query": "What variants in RARS1 are associated with hypomyelination?",
            "type": "expected_valid"
        },
        {
            "query": "Is RARS1 associated with Alzheimer's disease?",
            "type": "trick_question"
        }
    ]

    results = []

    for test in test_queries:

        query = test["query"]

        print(f"\nRunning query: {query}")

        try:
            answer = run_query(query)

        except Exception as e:
            answer = f"ERROR: {str(e)}"

        results.append({
            "query": query,
            "type": test["type"],
            "result": answer
        })

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nEvaluation complete. Results saved to eval_results.json")


if __name__ == "__main__":
    run_evaluation()