"""
cli.py
Interactive terminal interface for @AppleSupport AI Agent.
Allows evaluators and engineers to test individual queries either via command-line arguments
or through an interactive REPL terminal prompt.
"""

import sys
import argparse
from src.agent.pipeline import SupportAgentPipeline
from run_evaluation import load_data, prepare_train_labels

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def print_result_card(res):
    print("\n" + "=" * 70)
    print(f"[INPUT TWEET]       : {res.customer_text}")
    print(f"[PREDICTED INTENT]  : {res.intent} (Confidence: {res.confidence * 100:.1f}%)")
    
    # Escalation badge formatting
    status = "ESCALATE TO HUMAN" if res.should_escalate else "AUTO-HANDLE"
    print(f"[DECISION]          : [{status}] (Risk Level: {res.risk_level})")
    print(f"[STATED REASON]     : {res.escalation_reason}")
    
    if res.retrieved_resolutions:
        top_hist = res.retrieved_resolutions[0]
        print(f"[HISTORICAL FIX]    : (Match score: {top_hist.get('similarity_score', 0):.2f})")
        print(f"                      \"{top_hist.get('historical_resolution', '')[:90]}...\"")
        
    print(f"\n[DRAFTED REPLY]     : ({len(res.drafted_reply)} chars, Twitter limit: 280)")
    print(f"                      \"{res.drafted_reply}\"")
    print("=" * 70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="@AppleSupport AI Agent Terminal CLI")
    parser.add_argument("query", nargs="*", help="Optional customer tweet to process directly")
    args = parser.parse_args()

    print("Initializing @AppleSupport AI Agent pipeline...")
    golden_data, train_df = load_data()
    train_texts, train_labels = prepare_train_labels(train_df, golden_data)

    agent = SupportAgentPipeline(mode="production")
    agent.train_classifier(train_texts, train_labels)
    agent.retriever.load_and_index()
    print("Ready!\n")

    # If query was passed via command line arguments
    if args.query:
        query_text = " ".join(args.query)
        result = agent.process_message(query_text)
        print_result_card(result)
        return

    # Interactive REPL mode
    print("=" * 70)
    print(f"{'@AppleSupport AI Agent — Interactive Terminal Mode':^70}")
    print("=" * 70)
    print("Type any customer tweet or product inquiry below.")
    print("Type 'sample' for examples, or 'q' to quit.\n")

    samples = [
        "My iPhone 7 battery percentage jumps from 40% to 1% in five minutes.",
        "Someone hacked into my Apple ID and changed my trusted phone numbers. Help!",
        "I was charged $49.99 on iTunes that I never authorized! Refund please.",
        "My phone won't charge unless I bend the lightning cable at a 90-degree angle.",
        "How do I transfer contacts from my old iPhone 6 to my new iPhone 8?",
        "Worst customer service ever! Tim Cook is a criminal and I am suing you guys.",
        "Can you help me install a custom ROM on my Samsung Galaxy S8?"
    ]

    while True:
        try:
            user_input = input("Enter tweet > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["q", "quit", "exit"]:
                print("Exiting CLI. Goodbye!")
                break
            if user_input.lower() == "sample":
                print("\nSample queries to try (type the number or paste your text):")
                for i, s in enumerate(samples, 1):
                    print(f"  {i}. {s}")
                print()
                continue

            # If user entered a sample number (e.g. '1', '2', etc.)
            if user_input.isdigit() and 1 <= int(user_input) <= len(samples):
                sample_idx = int(user_input)
                user_input = samples[sample_idx - 1]
                print(f"\n[Selected Sample #{sample_idx}]: \"{user_input}\"")

            result = agent.process_message(user_input)
            print_result_card(result)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting CLI. Goodbye!")
            break

if __name__ == "__main__":
    main()
