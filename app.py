import gradio as gr
import pandas as pd

from src.inference.recommender import Recommender

# Load unified recommender once at startup
recommender = Recommender(
    models_dir="models",
    data_dir="data/processed",
)

STRATEGY_DESCRIPTIONS = {
    "hybrid": (
        "**Hybrid Recommendation Engine** — Combines 40% Content-Based (TF-IDF), "
        "40% BPR Collaborative Filtering, and 20% Popularity with automatic seen-item filtering."
    ),
    "cold_user_popularity": (
        "**Cold-User Fallback** — User is not present in the warm-user model intersection, "
        "so recommendations fall back to global popularity."
    ),
    "cold_item_category": (
        "**Cold-Item Category Recommendation** — Ranks unseen/cold items by category overlap "
        "with the user's learned category preferences."
    ),
}


def get_recommendations(user_id: int, k: int, mode: str, exclude_str: str):
    if user_id is None or user_id < 0:
        return "Please enter a valid non-negative User ID.", pd.DataFrame(columns=["Rank", "Item ID"])

    user_id = int(user_id)
    k = int(k)

    if mode == "Cold Items":
        result = recommender.recommend_cold_items(user_id=user_id, k=k)
    else:
        exclude_items = set()
        if exclude_str and exclude_str.strip():
            try:
                exclude_items = {
                    int(item.strip())
                    for item in exclude_str.split(",")
                    if item.strip()
                }
            except ValueError:
                return (
                    "**Error:** Excluded Item IDs must be comma-separated integers (e.g. `101, 202`).",
                    pd.DataFrame(columns=["Rank", "Item ID"]),
                )
        result = recommender.recommend(user_id=user_id, k=k, exclude_items=exclude_items)

    strategy = result["strategy"]
    items = result["recommendations"]
    desc = STRATEGY_DESCRIPTIONS.get(strategy, f"Strategy: `{strategy}`")

    status_md = (
        f"### User `{result['user_id']}` · Strategy: `{strategy}` · Returned `{len(items)}` items\n\n"
        f"{desc}"
    )

    df = pd.DataFrame(
        [{"Rank": rank, "Item ID": item_id} for rank, item_id in enumerate(items, start=1)]
    )
    if df.empty:
        df = pd.DataFrame(columns=["Rank", "Item ID"])

    return status_md, df


with gr.Blocks(title="Hybrid Recommendation Engine") as demo:
    gr.Markdown(
        "# 🎯 Hybrid Recommendation Engine\n"
        "Personalized recommendations powered by **Content-Based Filtering (TF-IDF)**, "
        "**BPR Collaborative Filtering**, **Popularity Ranking**, and **Cold-Start Routing**."
    )

    with gr.Row():
        with gr.Column(scale=1):
            user_id_input = gr.Number(label="User ID", value=172, precision=0, minimum=0)
            k_slider = gr.Slider(label="Number of Recommendations (K)", minimum=5, maximum=20, value=10, step=1)
            mode_radio = gr.Radio(
                label="Recommendation Mode",
                choices=["Personalized", "Cold Items"],
                value="Personalized",
            )
            exclude_input = gr.Textbox(
                label="Exclude Item IDs (optional, Personalized mode only)",
                placeholder="e.g. 12345, 67890",
            )
            submit_btn = gr.Button("Get Recommendations", variant="primary")

            gr.Examples(
                examples=[
                    [172, 10, "Personalized", ""],
                    [999999999, 10, "Personalized", ""],
                    [3, 10, "Cold Items", ""],
                ],
                inputs=[user_id_input, k_slider, mode_radio, exclude_input],
                label="Try Example Users (Warm User / Cold User / Cold Items)",
            )

        with gr.Column(scale=2):
            strategy_output = gr.Markdown("Click **Get Recommendations** to see results.")
            results_table = gr.Dataframe(
                headers=["Rank", "Item ID"],
                datatype=["number", "number"],
                interactive=False,
                label="Recommended Items",
            )

    submit_btn.click(
        fn=get_recommendations,
        inputs=[user_id_input, k_slider, mode_radio, exclude_input],
        outputs=[strategy_output, results_table],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
