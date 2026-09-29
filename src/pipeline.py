from detector import run_detection
from vlm_analyzer import VLMAnalyzer


def run_pipeline():

    print("\n==============================")
    print(" VisionGuard AI Pipeline")
    print("==============================")

    print(
        "\n[1/2] Running detection..."
    )

    events_dir = run_detection()

    print(
        "\n[2/2] Running VLM analysis..."
    )

    analyzer = VLMAnalyzer()

    analyzer.analyze_all(
        events_dir,
        skip_existing=True,
    )

    print(
        "\nPipeline complete."
    )

    return events_dir


def main():
    run_pipeline()


if __name__ == "__main__":
    main()