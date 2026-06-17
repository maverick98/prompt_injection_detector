from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from prompt_injection_detector.data.io import stratified_split
from prompt_injection_detector.data.synthetic import generate_synthetic_dataset
from prompt_injection_detector.models.classical import load_detector, train_classical_models
from prompt_injection_detector.redteam.strategies import RuleBasedRedTeamGenerator, score_variants


@st.cache_resource
def get_detector():
    model_path = Path("artifacts/detector.joblib")
    if model_path.exists():
        return load_detector(model_path)
    frame = pd.DataFrame([sample.model_dump(mode="json") for sample in generate_synthetic_dataset(300, 300)])
    frame = stratified_split(frame)
    return train_classical_models(frame[frame["split"] == "train"], frame[frame["split"] == "val"])


st.set_page_config(page_title="Prompt Injection Detector", layout="wide")
st.title("Prompt Injection Detector")

detector = get_detector()

left, right = st.columns(2)

with left:
    st.subheader("Detector")
    prompt = st.text_area(
        "Prompt input",
        height=220,
        placeholder="Paste a user prompt, document excerpt, or agent input here.",
    )
    if st.button("Analyze", type="primary") and prompt.strip():
        prediction = detector.predict_one(prompt)
        status = "Injection Detected" if prediction.is_injection else "Clean"
        st.metric(status, f"{prediction.score:.3f}")
        st.write(f"Category: `{prediction.category.value}`")
        if prediction.top_features:
            st.write("Top feature signals")
            st.dataframe(
                pd.DataFrame(prediction.top_features, columns=["feature", "weight"]),
                use_container_width=True,
            )

with right:
    st.subheader("Red-Team View")
    source = st.text_area(
        "Injection to mutate",
        value=prompt if "prompt" in locals() else "",
        height=160,
    )
    if st.button("Generate Evasions") and source.strip():
        variants = score_variants(detector, RuleBasedRedTeamGenerator().generate(source))
        st.dataframe(
            pd.DataFrame([variant.model_dump() for variant in variants])[
                ["strategy", "score", "bypassed", "variant_text"]
            ],
            use_container_width=True,
        )

