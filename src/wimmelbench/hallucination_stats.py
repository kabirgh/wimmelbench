from collections import defaultdict
import json
import os
from typing import Mapping

import matplotlib.pyplot as plt

from wimmelbench.stats import MODEL_DIRS


def plot_hallucination_stats(
    status_by_model: Mapping[str, Mapping[str, str]],
    output_path: str,
) -> None:
    """Create a horizontal bar chart showing hallucination rate for each model."""
    # Calculate percentages for each model
    model_stats = {}
    for model, statuses in status_by_model.items():
        total = len(statuses)
        hallucinated = sum(
            1 for status in statuses.values() if status == "hallucinated"
        )
        model_stats[model] = (hallucinated / total) * 100

    # Prepare data for plotting
    models = list(model_stats.keys())
    hallucinated_pcts = [model_stats[model] for model in models]

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Create horizontal bars
    bars = ax.barh(models, hallucinated_pcts, color="#234f81")

    # Add percentage labels
    for rect in bars:
        width = rect.get_width()
        if width > 0:  # Only show non-zero values
            x = rect.get_x() + width / 2
            y = rect.get_y() + rect.get_height() / 2
            ax.text(x, y, f"{width:.0f}%", ha="center", va="center", color="white")

    # Customize the chart
    ax.set_title("Hallucination rate by model\n(lower is better)", pad=20)
    ax.set_ylabel("Model")
    ax.set_xlabel("Percentage")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    # { model: { image: status } }
    status_by_model = defaultdict(lambda: defaultdict(str))

    for model_name, results_dir in MODEL_DIRS.items():
        results_path = os.path.join(results_dir, "results_telescope.json")
        with open(results_path, "r") as f:
            results_data = json.load(f)

        for img_id, img_data in results_data.items():
            status_by_model[model_name][img_id] = (
                "abstained"
                if img_data["telescope"]["bbox"] == [0, 0, 0, 0]
                else "hallucinated"
            )

    plot_hallucination_stats(status_by_model, "stats/hallucination_rates.png")


if __name__ == "__main__":
    main()
