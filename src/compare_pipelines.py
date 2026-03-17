from pathlib import Path
import pandas as pd

BASELINE_CSV = Path("outputs/metrics/baseline_detailed_results.csv")
PADDLE_CSV = Path("outputs/metrics/paddle_baseline_detailed_results.csv")
HYBRID_CSV = Path("outputs/metrics/hybrid_baseline_detailed_results.csv")

OUTPUT_CSV = Path("outputs/metrics/pipeline_group_comparison.csv")


def load_with_label(path: Path, pipeline_name: str) -> pd.DataFrame:
    if not path.exists():
        print(f"WARNING: file not found for {pipeline_name} -> {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        print(f"WARNING: file is empty for {pipeline_name} -> {path}")
        return df

    df = df.copy()
    df["pipeline"] = pipeline_name
    return df


def main():
    baseline_df = load_with_label(BASELINE_CSV, "tesseract_baseline")
    paddle_df = load_with_label(PADDLE_CSV, "paddle_baseline")
    hybrid_df = load_with_label(HYBRID_CSV, "hybrid_baseline")

    all_dfs = [df for df in [baseline_df, paddle_df, hybrid_df] if not df.empty]
    if not all_dfs:
        print("ERROR: no result files could be loaded.")
        return

    df = pd.concat(all_dfs, ignore_index=True)

    # Sanity check
    print("\n=== AVAILABLE GROUPS AND PIPELINES ===")
    print("Groups:", sorted(df["group"].unique().tolist()))
    print("Pipelines:", sorted(df["pipeline"].unique().tolist()))

    # Group + pipeline summary
    summary = (
        df.groupby(["group", "pipeline"])[["cer", "wer", "accuracy"]]
        .mean()
        .reset_index()
        .sort_values(["group", "accuracy"], ascending=[True, False])
    )

    print("\n=== GROUP x PIPELINE SUMMARY (sorted by group, then accuracy desc) ===")
    print(summary.to_string(index=False))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved summary to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()