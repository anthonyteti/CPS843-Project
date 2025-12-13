import json
from pathlib import Path


RUNS = [
    "small_aug",
    "medium_aug",
    "large_aug",
    "small_noaug",
    "medium_noaug",
    "large_noaug",
]


def show_json(path: Path) -> None:
    if not path.exists():
        print(f"!! Missing {path}")
        return
    print(f"\n== {path.as_posix()} ==")
    data = json.loads(path.read_text())
    print(json.dumps(data, indent=2))


def main() -> None:
    show_json(Path("results/classical/hog_svm_metrics.json"))
    for run in RUNS:
        metrics_path = Path("results/plots") / run / f"{run.split('_')[0]}_metrics.json"
        show_json(metrics_path)
    for run in RUNS:
        eval_path = Path("results/cnn_eval") / run / "cnn_eval_metrics.json"
        show_json(eval_path)


if __name__ == "__main__":
    main()
