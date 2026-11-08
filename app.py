"""Interactive research dashboard. Start with: python -m streamlit run app.py."""

import os
from pathlib import Path

import streamlit as st

from f1_forecast.reporting import aggregate_summary, blend_predictions, load_reports

ROOT = Path(__file__).resolve().parent


def main():
    st.set_page_config(page_title="F1 Race Lab", page_icon="🏁", layout="wide")
    st.markdown("### 🏁 F1 / RACE LAB")
    directory = Path(os.environ.get("F1_REPORT_DIR", str(ROOT / "reports/demo")))
    try:
        manifest, predictions, summary, calibration = load_reports(directory)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Could not load a complete report: {error}")
        st.code("python -m f1_forecast run --output artifacts")
        st.info("Then set F1_REPORT_DIR to the artifacts directory and restart the dashboard.")
        st.stop()

    race = manifest["race"]
    st.title(race["name"])
    st.caption(
        f"Pre-race research snapshot · {race['date']} · Training outcomes strictly before "
        f"{manifest['cutoff_exclusive']}"
    )
    st.warning(
        "Uncalibrated model scores: normalization does not establish real-world win odds. "
        "The bundled inputs are inherited research data and have not been independently verified."
    )

    with st.sidebar:
        st.header("Explore the forecast")
        weight = st.slider("V1 ensemble weight", 0.0, 1.0, 0.5, 0.05, key="v1_weight")
        st.caption(
            f"V1 {weight:.0%} / V2 {1 - weight:.0%}. The backtested ensemble stays fixed at 50/50."
        )
        top_n = st.slider("Drivers in chart", 1, len(predictions), min(10, len(predictions)))
        include_diagnostic = st.checkbox("Include diagnostic seasons", key="include_diagnostic")
        st.divider()
        st.caption(
            "V1: grid and recent form\n\nV2: adds qualifying gaps\n\nV3: experimental grid-displacement candidate"
        )
        st.caption(
            "Reports: "
            + (
                "Bundled demo"
                if directory.resolve() == (ROOT / "reports/demo").resolve()
                else str(directory)
            )
        )

    ranked = blend_predictions(predictions, weight)
    leader = ranked.iloc[0]
    cols = st.columns(4)
    cols[0].metric("Ensemble selection", leader.driver)
    cols[1].metric("Normalized ensemble score", f"{100 * leader.ensemble_score:.1f}%")
    cols[2].metric(
        "Starting position",
        "Pit lane" if leader.pit_lane_start else f"P{int(leader.grid_position)}",
    )
    cols[3].metric("Historical training races", manifest["training_races"])

    forecast_tab, evaluation_tab, risk_tab, method_tab = st.tabs(
        [
            "Race forecast",
            "Backtesting",
            "Grid risk",
            "Data & method",
        ]
    )
    with forecast_tab:
        st.subheader("The field, ranked by model score")
        chart = ranked.head(top_n).set_index("driver")[["ensemble_score"]] * 100
        st.bar_chart(
            chart.rename(columns={"ensemble_score": "Ensemble score (%)"}),
            horizontal=True,
            color="#ff4b4b",
        )
        table = ranked[
            ["driver", "team", "grid_position", "qualifying_position", "pit_lane_start"]
        ].copy()
        table.columns = ["Driver", "Team", "Grid", "Qualifying", "Pit lane"]
        for name in ("V1", "V2", "V3"):
            table[f"{name} score (%)"] = ranked[name] * 100
        table["Ensemble score (%)"] = ranked.ensemble_score * 100
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.download_button(
            "Download this forecast",
            table.to_csv(index=False).encode("utf-8"),
            file_name="f1_forecast.csv",
            mime="text/csv",
        )
        st.caption(
            "Changing ensemble weights is an exploratory scenario, not a validated model improvement. "
            "V3 is shown separately and is not part of the ensemble."
        )

    with evaluation_tab:
        st.subheader("Can the model beat picking the pole-sitter?")
        selected = summary if include_diagnostic else summary.loc[summary["sample"] == "historical"]
        if selected.empty:
            st.info(
                "No historical evaluation seasons in this config; enable diagnostics to inspect those folds."
            )
        else:
            aggregate = aggregate_summary(selected)
            st.bar_chart(aggregate[["Winner accuracy (%)"]], color="#ff4b4b")
            # Season-level results precede aggregate results to expose variation across years.
            st.dataframe(selected, hide_index=True, use_container_width=True)
            st.dataframe(aggregate, use_container_width=True)
        st.caption(
            "Each fold trains on earlier seasons only. Recent-form inputs update before each race; "
            "the fitted model stays fixed for the held-out season. Lower log loss/Brier is better."
        )
        boundary = manifest["evaluation"]["diagnostic_from"]
        st.info(
            f"Seasons from {boundary} onward have already informed research diagnostics. "
            "They are excluded by default and must not be presented as untouched holdouts."
        )
        if include_diagnostic:
            st.warning(
                "You are viewing historical and previously inspected diagnostic seasons together."
            )

    with risk_tab:
        st.subheader("Where grid position exposes model risk")
        model = st.selectbox("Calibration model", ["Blend_50_50", "V1", "V2", "V3"])
        sample_options = ["historical", "diagnostic"] if include_diagnostic else ["historical"]
        sample = st.selectbox("Calibration sample", sample_options)
        bands = calibration.loc[(calibration.model == model) & (calibration["sample"] == sample)]
        if bands.empty:
            st.info("No rows for this calibration sample.")
        else:
            st.bar_chart(bands.set_index("grid_band")[["predicted_wins", "actual_wins"]])
            st.dataframe(bands, hide_index=True, use_container_width=True)
        st.caption(
            "Predicted wins are sums of normalized scores over driver-race rows. "
            "Compare with observed wins to inspect aggregate calibration; small bands are noisy."
        )
        displaced = ranked.loc[
            ranked.grid_places_lost_vs_qualifying.ge(5),
            ["driver", "grid_position", "qualifying_position", "ensemble_score"],
        ]
        st.markdown(
            "**Target drivers starting at least five places behind their qualifying position**"
        )
        st.dataframe(displaced, hide_index=True, use_container_width=True)
        st.caption(
            "Grid displacement is not an official penalty count: other drivers' penalties and "
            "pit-lane starts can also move a driver."
        )

    with method_tab:
        st.subheader("A reproducible research snapshot")
        st.write(manifest["provenance"])
        st.write(race.get("snapshot_status", "User-supplied pre-race grid"))
        st.markdown(
            "**Workflow:** historical results → lagged form and qualifying gaps → "
            "season backtests → final training → pre-race normalized scores."
        )
        st.markdown(
            "Driver form uses the previous five starts. Team form uses team points over the "
            "previous five races. Missing history is imputed using training data only. "
            "Wet qualifying, strategy, incidents and regulation changes can limit transfer."
        )
        st.code("python -m f1_forecast run --config config/sao_paulo_2026.toml --output artifacts")
        with st.expander("Input hashes, model features and software versions"):
            st.json(manifest)
        st.caption(
            "Research and educational use. This dashboard is not a live timing or betting service."
        )


if __name__ == "__main__":
    main()
