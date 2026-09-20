import sys

from src.pipeline.pipeline import run_research_pipeline


def main() -> None:
    topic = " ".join(sys.argv[1:]).strip() or input("Enter a research topic: ").strip()
    if not topic:
        print("No topic provided.")
        sys.exit(1)

    state = run_research_pipeline(topic)

    out_file = "report.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"# {topic}\n\n{state['report']}\n\n---\n\n## Critic Feedback\n\n{state['feedback']}\n")
    print(f"\nReport saved to {out_file}")


if __name__ == "__main__":
    main()
