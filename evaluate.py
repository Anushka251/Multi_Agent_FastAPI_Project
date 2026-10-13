
import json
from pathlib import Path

import nltk
import requests
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

API_URL = "http://127.0.0.1:8000/query"
OUTPUT_DIR = Path("evaluation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Questions and reference answers for the four FAQ documents.
TEST_CASES = [
    {
        "question": "How do I track my order?",
        "reference": "Use the tracking link in your email. Contact support with your order ID if the link does not work.",
        "baseline": "You can contact customer support for help with your order."
    },
    {
        "question": "Can I return an item?",
        "reference": "Returns are accepted within 7 days if the conditions in the return policy are met. Provide your order ID and reason.",
        "baseline": "Check the website for information about returns."
    },
    {
        "question": "When will I receive my refund?",
        "reference": "Refunds are processed after the returned item is received and inspected. The refund goes to the original payment method, and timing depends on the payment provider.",
        "baseline": "Refunds may take some time to arrive."
    },
    {
        "question": "I cannot access my account. What should I do?",
        "reference": "Use Forgot Password and follow the instructions in the email. Check your spam folder if the email does not arrive.",
        "baseline": "Try logging in again or contact support."
    }
]

def find_answer(data):
    """Extract an answer from common API response formats."""
    if isinstance(data, str):
        return data

    if isinstance(data, dict):
        for key in ("answer", "response", "output", "result"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value

        for value in data.values():
            if isinstance(value, (dict, list)):
                answer = find_answer(value)
                if answer:
                    return answer

    if isinstance(data, list):
        for item in data:
            answer = find_answer(item)
            if answer:
                return answer

    return ""

def get_agent_answer(question):
    response = requests.post(
        API_URL,
        json={"query": question},
        timeout=120,
    )
    response.raise_for_status()
    answer = find_answer(response.json())

    if not answer:
        raise ValueError(
            f"Could not find the answer in API response: {response.text}"
        )

    return answer

def calculate_metrics(prediction, reference, embedding_model, scorer):
    prediction = prediction.strip()
    reference = reference.strip()

    bleu = sentence_bleu(
        [reference.lower().split()],
        prediction.lower().split(),
        smoothing_function=SmoothingFunction().method1,
    )

    rouge = scorer.score(reference, prediction)

    vectors = embedding_model.encode(
        [reference, prediction],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    relevance = float(util.cos_sim(vectors[0], vectors[1]).item())

    return {
        "BLEU": round(bleu, 4),
        "ROUGE-1 F1": round(rouge["rouge1"].fmeasure, 4),
        "ROUGE-2 F1": round(rouge["rouge2"].fmeasure, 4),
        "ROUGE-L F1": round(rouge["rougeL"].fmeasure, 4),
        "Semantic similarity": round(relevance, 4),
    }

def main():
    print("Loading evaluation model...")
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    rows = []

    for case in TEST_CASES:
        question = case["question"]
        reference = case["reference"]

        print(f"\nEvaluating: {question}")
        agent_answer = get_agent_answer(question)

        print("Baseline:", case["baseline"])
        print("Current graph:", agent_answer)

        baseline_metrics = calculate_metrics(
            case["baseline"], reference, embedding_model, scorer
        )
        graph_metrics = calculate_metrics(
            agent_answer, reference, embedding_model, scorer
        )

        for system_name, answer, metrics in [
            ("Baseline", case["baseline"], baseline_metrics),
            ("Current graph (not yet RAG)", agent_answer, graph_metrics),
        ]:
            rows.append({
                "Question": question,
                "System": system_name,
                "Reference": reference,
                "Answer": answer,
                **metrics,
            })

    # Save detailed results as JSON.
    (OUTPUT_DIR / "evaluation_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Save a readable CSV table.
    import csv

    with (OUTPUT_DIR / "evaluation_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Print average scores for the comparison table.
    metric_names = [
        "BLEU", "ROUGE-1 F1", "ROUGE-2 F1",
        "ROUGE-L F1", "Semantic similarity"
    ]

    print("\nAVERAGE EVALUATION SCORES")
    print("=" * 65)

    for system_name in ("Baseline", "Current graph (not yet RAG)"):
        group = [r for r in rows if r["System"] == system_name]
        averages = {
            metric: sum(r[metric] for r in group) / len(group)
            for metric in metric_names
        }
        print(f"\n{system_name}")
        for metric, value in averages.items():
            print(f"  {metric}: {value:.4f}")

    print("\nResults saved to:")
    print((OUTPUT_DIR / "evaluation_results.csv").resolve())
    print((OUTPUT_DIR / "evaluation_results.json").resolve())

if __name__ == "__main__":
    main()
