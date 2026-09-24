"""
AI-Based Sentiment Analysis System — Product Sentiment Analytics Dashboard
Production-grade multi-page Streamlit application using the Phase 4 Benchmark Model:
TF-IDF + 12 Linguistic Features + Logistic Regression (Macro F1 = 0.6657, Full 102,076 training samples).

Strictly inference only. No training routines are called.
"""

import sys
import io
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st

# ── Paths & Environment ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Reuse exact Phase 4 feature extraction & text preprocessing
try:
    from src.linguistic_features import extract_features_from_text, FEATURE_NAMES
    from src.text_preprocessing import preprocess_for_tfidf
except ImportError as err:
    st.error(f"Critical Error: Unable to import modules from src/: {err}")
    st.stop()

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Product Sentiment Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Modern SaaS Dashboard Aesthetics ───────────────────────────
st.markdown(
    """
    <style>
    /* Clean modern SaaS styling */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        text-align: center;
        margin-bottom: 0.8rem;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0F172A;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        color: #FFFFFF;
    }
    .badge-pos { background-color: #10B981; }
    .badge-neu { background-color: #3B82F6; }
    .badge-neg { background-color: #EF4444; }
    
    .footer-bar {
        margin-top: 3.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid #E2E8F0;
        text-align: center;
        color: #94A3B8;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Model Artifacts Loader (Cached & Read-Only) ────────────────────────────────
@st.cache_resource(show_spinner="Loading trained Phase 4 production model...")
def load_production_model():
    """
    Loads pre-trained Phase 4 artifacts.
    Strictly read-only; no training functions exist here.
    """
    tfidf_file = MODELS_DIR / "tfidf_vectorizer.pkl"
    scaler_file = MODELS_DIR / "linguistic_feature_scaler.pkl"
    model_file = MODELS_DIR / "logistic_regression_linguistic.pkl"

    missing = []
    for p, name in [
        (tfidf_file, "TF-IDF Vectorizer"),
        (scaler_file, "Linguistic Feature Scaler"),
        (model_file, "Logistic Regression Model"),
    ]:
        if not p.exists():
            missing.append(f"{name} ({p.name})")

    if missing:
        return None, None, None, f"Missing required model artifact(s): {', '.join(missing)}"

    try:
        with open(tfidf_file, "rb") as f:
            tfidf = pickle.load(f)
        with open(scaler_file, "rb") as f:
            scaler = pickle.load(f)
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        return tfidf, scaler, model, None
    except Exception as exc:
        return None, None, None, f"Failed to load artifacts: {str(exc)}"


tfidf_vec, feat_scaler, clf_model, load_err = load_production_model()

# ── Batch & Single Inference Functions ────────────────────────────────────────
def predict_single(text: str):
    """Predict sentiment and probabilities for a single text."""
    clean_text = preprocess_for_tfidf(text)
    tfidf_mat = tfidf_vec.transform([clean_text])
    ling_dict = extract_features_from_text(text)
    ling_vec = np.array([[ling_dict[name] for name in FEATURE_NAMES]])
    scaled_ling = feat_scaler.transform(ling_vec)
    X_comb = sp.hstack([tfidf_mat, sp.csr_matrix(scaled_ling)], format="csr")

    pred = clf_model.predict(X_comb)[0]
    probas = clf_model.predict_proba(X_comb)[0]
    classes = list(clf_model.classes_)
    prob_dict = {c: float(p) for c, p in zip(classes, probas)}

    conf = max(probas)
    return {
        "prediction": pred,
        "confidence": float(conf),
        "negative_prob": prob_dict.get("negative", 0.0),
        "neutral_prob": prob_dict.get("neutral", 0.0),
        "positive_prob": prob_dict.get("positive", 0.0),
        "features": ling_dict,
    }


def predict_batch_df(df: pd.DataFrame, text_col: str):
    """
    Run vectorized batch inference on a DataFrame column without modifying original columns.
    Returns a copy of the DataFrame with added prediction columns.
    """
    texts = df[text_col].fillna("").astype(str).tolist()
    clean_texts = [preprocess_for_tfidf(t) for t in texts]
    tfidf_mat = tfidf_vec.transform(clean_texts)

    feat_rows = []
    for t in texts:
        f_dict = extract_features_from_text(t)
        feat_rows.append([f_dict[col] for col in FEATURE_NAMES])
    feat_mat = np.array(feat_rows)
    scaled_feat = feat_scaler.transform(feat_mat)

    X_comb = sp.hstack([tfidf_mat, sp.csr_matrix(scaled_feat)], format="csr")
    preds = clf_model.predict(X_comb)
    probas = clf_model.predict_proba(X_comb)

    classes = list(clf_model.classes_)
    neg_idx = classes.index("negative") if "negative" in classes else 0
    neu_idx = classes.index("neutral") if "neutral" in classes else 1
    pos_idx = classes.index("positive") if "positive" in classes else 2

    out_df = df.copy()
    out_df["predicted_sentiment"] = preds
    out_df["confidence"] = np.round(np.max(probas, axis=1), 4)
    out_df["negative_probability"] = np.round(probas[:, neg_idx], 4)
    out_df["neutral_probability"] = np.round(probas[:, neu_idx], 4)
    out_df["positive_probability"] = np.round(probas[:, pos_idx], 4)
    return out_df


# ── Built-in Realistic Demo Dataset ───────────────────────────────────────────
def get_sample_reviews_df():
    """Generates a realistic multi-product e-commerce dataset for instant demo exploration."""
    sample_records = [
        {"product_name": "EchoPulse Pro Headphones", "date": "2026-08-01", "review_text": "I absolutely loved this movie and the audio fidelity is breathtaking!"},
        {"product_name": "EchoPulse Pro Headphones", "date": "2026-08-03", "review_text": "Sound is very clear and the noise cancellation works like magic."},
        {"product_name": "EchoPulse Pro Headphones", "date": "2026-08-05", "review_text": "This product is terrible and disappointing. The left ear cup stopped working."},
        {"product_name": "EchoPulse Pro Headphones", "date": "2026-08-07", "review_text": "The package arrived today. Standard packaging with black case."},
        {"product_name": "EchoPulse Pro Headphones", "date": "2026-08-10", "review_text": "Battery life lasts over 30 hours, absolutely fantastic purchase!"},
        {"product_name": "EchoPulse Pro Headphones", "date": "2026-08-12", "review_text": "Average sound quality. Nothing special but acceptable for the price."},
        {"product_name": "AuraFit Ultra Smartwatch", "date": "2026-08-02", "review_text": "The service was excellent and very helpful with setup. Great battery!"},
        {"product_name": "AuraFit Ultra Smartwatch", "date": "2026-08-04", "review_text": "Heart rate monitor is accurate and the screen is bright in sunlight."},
        {"product_name": "AuraFit Ultra Smartwatch", "date": "2026-08-06", "review_text": "I am not happy with the experience. The companion app keeps crashing."},
        {"product_name": "AuraFit Ultra Smartwatch", "date": "2026-08-08", "review_text": "The strap color is dark grey as shown in the specifications."},
        {"product_name": "AuraFit Ultra Smartwatch", "date": "2026-08-11", "review_text": "Step counter is decent but sleep tracking is occasionally off."},
        {"product_name": "AuraFit Ultra Smartwatch", "date": "2026-08-13", "review_text": "Completely defective sensor. Requested replacement immediately."},
        {"product_name": "NovaDesk Ergonomic Chair", "date": "2026-08-02", "review_text": "Exceptional lumbar support and very comfortable for 10-hour workdays."},
        {"product_name": "NovaDesk Ergonomic Chair", "date": "2026-08-05", "review_text": "Assembly manual had 6 steps and parts were clearly labeled."},
        {"product_name": "NovaDesk Ergonomic Chair", "date": "2026-08-07", "review_text": "Extremely squeaky after two weeks. Poor build quality."},
        {"product_name": "NovaDesk Ergonomic Chair", "date": "2026-08-09", "review_text": "Delivered on Monday afternoon via courier."},
        {"product_name": "NovaDesk Ergonomic Chair", "date": "2026-08-12", "review_text": "Sturdy base and smooth rolling wheels. Highly recommended!"},
        {"product_name": "Zenith 4K Gaming Monitor", "date": "2026-08-03", "review_text": "Brilliant colors, high refresh rate, and gorgeous HDR gaming visuals."},
        {"product_name": "Zenith 4K Gaming Monitor", "date": "2026-08-06", "review_text": "The package arrived today with an HDMI and DisplayPort cable."},
        {"product_name": "Zenith 4K Gaming Monitor", "date": "2026-08-09", "review_text": "Horrible backlight bleeding in every corner. Total waste of money."},
        {"product_name": "Zenith 4K Gaming Monitor", "date": "2026-08-11", "review_text": "Monitor stand is height adjustable and supports VESA mounting."},
        {"product_name": "Zenith 4K Gaming Monitor", "date": "2026-08-14", "review_text": "I was skeptical at first, but this monitor exceeded all my expectations!"},
    ]
    return pd.DataFrame(sample_records)


# Initialize session state for analyzed dataset
if "analyzed_df" not in st.session_state:
    if tfidf_vec is not None:
        raw_sample = get_sample_reviews_df()
        st.session_state["analyzed_df"] = predict_batch_df(raw_sample, "review_text")
    else:
        st.session_state["analyzed_df"] = None

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Product Analytics")
    st.markdown("*AI-Powered Sentiment Intelligence*")
    st.divider()

    nav_selection = st.radio(
        "Navigation Menu",
        [
            "Dashboard",
            "Single Review",
            "Bulk Analysis",
            "Product Analysis",
            "Review Explorer",
            "Analytics",
            "Experiments",
            "Model Information",
            "Export",
        ],
        index=0,
    )

    st.divider()
    st.markdown("**Production Model:** Phase 4 Benchmark")
    st.caption("TF-IDF + 12 Linguistic Features + Logistic Regression (Macro F1 = 0.6657)")
    st.divider()
    
    # Quick reload demo dataset button
    if st.button("🔄 Reset to Demo Dataset", use_container_width=True):
        if tfidf_vec is not None:
            raw_sample = get_sample_reviews_df()
            st.session_state["analyzed_df"] = predict_batch_df(raw_sample, "review_text")
            st.success("Loaded demo dataset with 22 reviews!")
            st.rerun()

# Check for load errors before rendering pages
if load_err:
    st.error(f"❌ {load_err}")
    st.info("Please verify that all `.pkl` files exist in the `models/` directory.")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# 1. DASHBOARD PAGE
# ──────────────────────────────────────────────────────────────────────────────
if nav_selection == "Dashboard":
    st.title("AI-Based Sentiment Analysis System")
    st.subheader("Three-Class Sentiment Analysis using TF-IDF, Linguistic Features & Logistic Regression")
    st.markdown("### 📈 Executive Sentiment Overview")

    df = st.session_state.get("analyzed_df")

    if df is None or len(df) == 0:
        st.info("No reviews currently analyzed. Head over to **Bulk Analysis** to upload reviews or click 'Reset to Demo Dataset' in the sidebar.")
    else:
        # Product filter if multiple products exist
        has_product = "product_name" in df.columns or "product" in df.columns
        prod_col = "product_name" if "product_name" in df.columns else ("product" if "product" in df.columns else None)

        display_df = df
        selected_prod = "All Products"
        if prod_col:
            products = ["All Products"] + sorted(df[prod_col].dropna().unique().tolist())
            selected_prod = st.selectbox("Select Product to Inspect:", products, index=0)
            if selected_prod != "All Products":
                display_df = df[df[prod_col] == selected_prod]

        total_count = len(display_df)
        pos_count = (display_df["predicted_sentiment"] == "positive").sum()
        neu_count = (display_df["predicted_sentiment"] == "neutral").sum()
        neg_count = (display_df["predicted_sentiment"] == "negative").sum()

        pos_pct = (pos_count / total_count * 100) if total_count > 0 else 0
        neu_pct = (neu_count / total_count * 100) if total_count > 0 else 0
        neg_pct = (neg_count / total_count * 100) if total_count > 0 else 0
        avg_conf = display_df["confidence"].mean() if total_count > 0 else 0.0

        # KPI Metrics Row
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Reviews</div><div class="metric-value">{total_count:,}</div></div>', unsafe_allow_html=True)
        with kpi2:
            st.markdown(f'<div class="metric-card"><div class="metric-label" style="color:#10B981;">Positive %</div><div class="metric-value" style="color:#10B981;">{pos_pct:.1f}%</div></div>', unsafe_allow_html=True)
        with kpi3:
            st.markdown(f'<div class="metric-card"><div class="metric-label" style="color:#3B82F6;">Neutral %</div><div class="metric-value" style="color:#3B82F6;">{neu_pct:.1f}%</div></div>', unsafe_allow_html=True)
        with kpi4:
            st.markdown(f'<div class="metric-card"><div class="metric-label" style="color:#EF4444;">Negative %</div><div class="metric-value" style="color:#EF4444;">{neg_pct:.1f}%</div></div>', unsafe_allow_html=True)
        with kpi5:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Confidence</div><div class="metric-value">{avg_conf * 100:.1f}%</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # Charts Row
        chart_col1, chart_col2 = st.columns([1, 1])

        with chart_col1:
            st.markdown("##### 📊 Sentiment Class Distribution")
            dist_data = pd.DataFrame({
                "Sentiment": ["Positive", "Neutral", "Negative"],
                "Count": [pos_count, neu_count, neg_count],
            }).set_index("Sentiment")
            st.bar_chart(dist_data, color="#3B82F6", use_container_width=True)

        with chart_col2:
            st.markdown("##### 📅 Sentiment Trend Over Time")
            has_date = "date" in display_df.columns
            if has_date:
                try:
                    trend_df = display_df.copy()
                    trend_df["date"] = pd.to_datetime(trend_df["date"])
                    trend_grouped = trend_df.groupby([trend_df["date"].dt.date, "predicted_sentiment"]).size().unstack(fill_value=0)
                    for col in ["positive", "neutral", "negative"]:
                        if col not in trend_grouped.columns:
                            trend_grouped[col] = 0
                    st.line_chart(trend_grouped[["positive", "neutral", "negative"]], use_container_width=True)
                except Exception:
                    st.info("Unable to parse date column into time series.")
            else:
                st.info("No date column present in dataset to plot trend.")

        st.markdown("---")
        st.markdown("##### 📝 Recent Analyzed Reviews")
        text_col_found = [c for c in display_df.columns if c in ["review_text", "review", "text", "comment", "feedback"]]
        preview_text_col = text_col_found[0] if text_col_found else display_df.columns[0]
        
        display_cols = [c for c in [prod_col, "date", preview_text_col, "predicted_sentiment", "confidence"] if c and c in display_df.columns]
        st.dataframe(display_df[display_cols].tail(10), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2. SINGLE REVIEW ANALYSIS PAGE
# ──────────────────────────────────────────────────────────────────────────────
elif nav_selection == "Single Review":
    st.title("Single Review Analysis")
    st.markdown("Test individual sentences or customer reviews using the production Phase 4 model.")

    if "single_input_text" not in st.session_state:
        st.session_state["single_input_text"] = ""

    def set_single_text(txt):
        st.session_state["single_input_text"] = txt

    st.markdown("**Quick Preset Examples:**")
    ex_c1, ex_c2, ex_c3 = st.columns(3)
    with ex_c1:
        if st.button("🟢 'I absolutely loved this movie!'", use_container_width=True):
            set_single_text("I absolutely loved this movie!")
    with ex_c2:
        if st.button("🔴 'This product is terrible and disappointing.'", use_container_width=True):
            set_single_text("This product is terrible and disappointing.")
    with ex_c3:
        if st.button("🔵 'The package arrived today.'", use_container_width=True):
            set_single_text("The package arrived today.")

    review_input = st.text_area(
        label="Enter review text to analyze:",
        value=st.session_state["single_input_text"],
        height=140,
        placeholder="Type a sentence such as: I absolutely loved this movie!",
        key="single_review_box",
    )

    btn_analyze = st.button("Analyze Sentiment", type="primary", use_container_width=True)

    if btn_analyze or (review_input and review_input.strip()):
        clean_text = review_input.strip()
        if not clean_text:
            if btn_analyze:
                st.warning("Please enter some text to analyze.")
        else:
            result = predict_single(clean_text)
            pred = result["prediction"]
            conf = result["confidence"]
            neg_p = result["negative_prob"]
            neu_p = result["neutral_prob"]
            pos_p = result["positive_prob"]

            st.markdown("---")
            st.markdown("### Prediction Results")

            badge_color = "badge-pos" if pred == "positive" else ("badge-neg" if pred == "negative" else "badge-neu")
            icon = "🟢" if pred == "positive" else ("🔴" if pred == "negative" else "🔵")

            rc1, rc2 = st.columns([1, 1])
            with rc1:
                st.markdown(f'<div class="badge-pill {badge_color}">{icon} PREDICTED: {pred.upper()}</div>', unsafe_allow_html=True)
            with rc2:
                st.metric("Confidence Score", f"{conf * 100:.2f}%")

            st.markdown("##### Sentiment Probabilities")
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                st.markdown(f'<div class="metric-card"><div class="metric-label" style="color:#EF4444;">Negative</div><div class="metric-value" style="color:#EF4444;">{neg_p * 100:.2f}%</div></div>', unsafe_allow_html=True)
                st.progress(neg_p)
            with pc2:
                st.markdown(f'<div class="metric-card"><div class="metric-label" style="color:#3B82F6;">Neutral</div><div class="metric-value" style="color:#3B82F6;">{neu_p * 100:.2f}%</div></div>', unsafe_allow_html=True)
                st.progress(neu_p)
            with pc3:
                st.markdown(f'<div class="metric-card"><div class="metric-label" style="color:#10B981;">Positive</div><div class="metric-value" style="color:#10B981;">{pos_p * 100:.2f}%</div></div>', unsafe_allow_html=True)
                st.progress(pos_p)

            # Linguistic feature breakdown
            with st.expander("🔍 Extracted 12 Linguistic Features Breakdown"):
                f = result["features"]
                c_a, c_b, c_c, c_d = st.columns(4)
                with c_a:
                    st.markdown("**Group A: Statistics**")
                    st.write(f"- Word Count: `{int(f['word_count'])}`")
                    st.write(f"- Char Count: `{int(f['character_count'])}`")
                    st.write(f"- Sent Length: `{int(f['sentence_length'])}`")
                with c_b:
                    st.markdown("**Group B: Syntax/Casing**")
                    st.write(f"- Exclamations: `{int(f['exclamation_count'])}`")
                    st.write(f"- Questions: `{int(f['question_count'])}`")
                    st.write(f"- Upper Ratio: `{f['uppercase_ratio']:.2f}`")
                    st.write(f"- Upper Words: `{int(f['uppercase_word_count'])}`")
                with c_c:
                    st.markdown("**Group C: Sentiment Cues**")
                    st.write(f"- Negations: `{int(f['negation_count'])}`")
                    st.write(f"- Intensifiers: `{int(f['intensifier_count'])}`")
                    st.write(f"- Contrast Words: `{int(f['contrast_word_count'])}`")
                with c_d:
                    st.markdown("**Group D: Visual/Repetition**")
                    st.write(f"- Emojis: `{int(f['emoji_count'])}`")
                    st.write(f"- Repeated Chars: `{int(f['repeated_character_count'])}`")


# ──────────────────────────────────────────────────────────────────────────────
# 3. BULK REVIEW ANALYSIS PAGE
# ──────────────────────────────────────────────────────────────────────────────
elif nav_selection == "Bulk Analysis":
    st.title("Bulk Review Analysis (CSV Upload)")
    st.markdown("Upload a CSV file containing reviews to analyze the entire dataset with the Phase 4 model.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            if raw_df.empty:
                st.warning("The uploaded CSV is empty. Please upload a CSV containing review rows.")
            else:
                st.success(f"Uploaded successfully: `{uploaded_file.name}` ({len(raw_df):,} rows)")

                # Identify review column
                candidate_cols = ["review", "review_text", "text", "comment", "feedback"]
                found_col = None
                for col in raw_df.columns:
                    if col.lower().strip() in candidate_cols:
                        found_col = col
                        break

                if found_col is None:
                    selected_col = st.selectbox("Select the column containing the review text:", raw_df.columns)
                else:
                    selected_col = st.selectbox("Review text column detected (or select another):", raw_df.columns, index=list(raw_df.columns).index(found_col))

                if st.button("🚀 Run Bulk Sentiment Analysis", type="primary", use_container_width=True):
                    with st.spinner(f"Analyzing {len(raw_df):,} reviews locally with Phase 4 model..."):
                        analyzed = predict_batch_df(raw_df, selected_col)
                        st.session_state["analyzed_df"] = analyzed
                        st.success("✅ Analysis complete! Data has been updated across all dashboard pages.")

                # If current analyzed data exists, display table & download button
                current_df = st.session_state.get("analyzed_df")
                if current_df is not None and "predicted_sentiment" in current_df.columns:
                    st.markdown("---")
                    st.markdown("### Analyzed Dataset Preview")
                    st.dataframe(current_df, use_container_width=True)

                    csv_buffer = current_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Analyzed CSV",
                        data=csv_buffer,
                        file_name=f"sentiment_analyzed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
        except Exception as e:
            st.error(f"Error parsing CSV file: {str(e)}")
    else:
        st.info("💡 You can upload your own CSV file with reviews, or explore the current dataset already populated in the dashboard.")


# ──────────────────────────────────────────────────────────────────────────────
# 4. PRODUCT ANALYSIS PAGE
# ──────────────────────────────────────────────────────────────────────────────
elif nav_selection == "Product Analysis":
    st.title("Product-Level Sentiment Analysis")
    st.markdown("Drill down into sentiment metrics and distribution for specific products.")

    df = st.session_state.get("analyzed_df")

    if df is None or len(df) == 0:
        st.info("No analyzed reviews available. Please upload a dataset in Bulk Analysis or reset to the demo dataset.")
    else:
        prod_col = "product_name" if "product_name" in df.columns else ("product" if "product" in df.columns else None)

        if prod_col is None:
            st.warning("The current dataset does not have a 'product_name' or 'product' column.")
            st.info("You can enter a custom product name below to label the whole dataset:")
            custom_prod = st.text_input("Product Name:", value="My Product")
            target_df = df
        else:
            product_list = sorted(df[prod_col].dropna().unique().tolist())
            selected_prod = st.selectbox("Select Product to Analyze:", product_list)
            target_df = df[df[prod_col] == selected_prod]

        prod_count = len(target_df)
        p_pos = (target_df["predicted_sentiment"] == "positive").sum()
        p_neu = (target_df["predicted_sentiment"] == "neutral").sum()
        p_neg = (target_df["predicted_sentiment"] == "negative").sum()

        pct_pos = (p_pos / prod_count * 100) if prod_count > 0 else 0
        pct_neu = (p_neu / prod_count * 100) if prod_count > 0 else 0
        pct_neg = (p_neg / prod_count * 100) if prod_count > 0 else 0
        p_conf = target_df["confidence"].mean() if prod_count > 0 else 0.0

        st.markdown("---")
        pk1, pk2, pk3, pk4, pk5 = st.columns(5)
        with pk1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Review Count</div><div class="metric-value">{prod_count:,}</div></div>', unsafe_allow_html=True)
        with pk2:
            st.markdown(f'<div class="metric-card"><div class="metric-label" style="color:#10B981;">Positive %</div><div class="metric-value" style="color:#10B981;">{pct_pos:.1f}%</div></div>', unsafe_allow_html=True)
        with pk3:
            st.markdown(f'<div class="metric-card"><div class="metric-label" style="color:#3B82F6;">Neutral %</div><div class="metric-value" style="color:#3B82F6;">{pct_neu:.1f}%</div></div>', unsafe_allow_html=True)
        with pk4:
            st.markdown(f'<div class="metric-card"><div class="metric-label" style="color:#EF4444;">Negative %</div><div class="metric-value" style="color:#EF4444;">{pct_neg:.1f}%</div></div>', unsafe_allow_html=True)
        with pk5:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Confidence</div><div class="metric-value">{p_conf * 100:.1f}%</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("##### Product Sentiment Distribution")
            chart_data = pd.DataFrame({
                "Sentiment": ["Positive", "Neutral", "Negative"],
                "Count": [p_pos, p_neu, p_neg],
            }).set_index("Sentiment")
            st.bar_chart(chart_data, use_container_width=True)

        with c2:
            st.markdown("##### Product Reviews Sample")
            txt_col = [c for c in target_df.columns if c in ["review_text", "review", "text", "comment", "feedback"]]
            t_col = txt_col[0] if txt_col else target_df.columns[0]
            st.dataframe(target_df[[t_col, "predicted_sentiment", "confidence"]].head(8), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# 5. REVIEW EXPLORER PAGE
# ──────────────────────────────────────────────────────────────────────────────
elif nav_selection == "Review Explorer":
    st.title("Review Explorer & Filter")
    st.markdown("Search, filter, and inspect individual review sentiment classifications and probabilities.")

    df = st.session_state.get("analyzed_df")

    if df is None or len(df) == 0:
        st.info("No reviews available to explore. Please upload a dataset in Bulk Analysis.")
    else:
        # Filter controls
        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            sent_filter = st.selectbox("Filter by Sentiment:", ["All", "positive", "neutral", "negative"], index=0)

        with f_col2:
            prod_col = "product_name" if "product_name" in df.columns else ("product" if "product" in df.columns else None)
            if prod_col:
                prod_opts = ["All"] + sorted(df[prod_col].dropna().unique().tolist())
                prod_filter = st.selectbox("Filter by Product:", prod_opts, index=0)
            else:
                prod_filter = "All"
                st.selectbox("Filter by Product:", ["All (No Product Col)"], disabled=True)

        with f_col3:
            min_conf = st.slider("Minimum Confidence:", min_value=0.0, max_value=1.0, value=0.50, step=0.05)

        search_query = st.text_input("🔍 Search Text (keywords in review):", placeholder="e.g., sound, battery, support...")

        # Apply filters
        filtered_df = df.copy()

        if sent_filter != "All":
            filtered_df = filtered_df[filtered_df["predicted_sentiment"] == sent_filter]

        if prod_col and prod_filter != "All":
            filtered_df = filtered_df[filtered_df[prod_col] == prod_filter]

        filtered_df = filtered_df[filtered_df["confidence"] >= min_conf]

        txt_col = [c for c in df.columns if c in ["review_text", "review", "text", "comment", "feedback"]]
        review_col = txt_col[0] if txt_col else df.columns[0]

        if search_query.strip():
            filtered_df = filtered_df[filtered_df[review_col].astype(str).str.contains(search_query.strip(), case=False, na=False)]

        st.markdown(f"**Found {len(filtered_df):,} matching reviews:**")
        st.dataframe(filtered_df, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# 6. ANALYTICS PAGE
# ──────────────────────────────────────────────────────────────────────────────
elif nav_selection == "Analytics":
    st.title("Sentiment Analytics & Insights")
    st.markdown("Detailed visual distributions, timelines, and confidence metrics.")

    df = st.session_state.get("analyzed_df")

    if df is None or len(df) == 0:
        st.info("No data available for analytics. Upload a dataset in Bulk Analysis.")
    else:
        an_col1, an_col2 = st.columns(2)

        with an_col1:
            st.markdown("##### 1. Sentiment Count Distribution")
            counts = df["predicted_sentiment"].value_axis = df["predicted_sentiment"].value_counts()
            st.bar_chart(counts, use_container_width=True)

        with an_col2:
            st.markdown("##### 2. Sentiment Percentages")
            pcts = (df["predicted_sentiment"].value_counts(normalize=True) * 100).round(2)
            pcts_df = pd.DataFrame({"Percentage (%)": pcts})
            st.dataframe(pcts_df, use_container_width=True)

        st.markdown("---")
        an_col3, an_col4 = st.columns(2)

        with an_col3:
            st.markdown("##### 3. Confidence Distribution (Histogram)")
            conf_bins = pd.cut(df["confidence"], bins=[0.3, 0.5, 0.7, 0.85, 1.0], labels=["0.3-0.5", "0.5-0.7", "0.7-0.85", "0.85-1.0"])
            conf_counts = conf_bins.value_counts().sort_index()
            st.bar_chart(conf_counts, use_container_width=True)

        with an_col4:
            st.markdown("##### 4. Sentiment by Product (if available)")
            prod_col = "product_name" if "product_name" in df.columns else ("product" if "product" in df.columns else None)
            if prod_col:
                prod_sent = df.groupby([prod_col, "predicted_sentiment"]).size().unstack(fill_value=0)
                st.bar_chart(prod_sent, use_container_width=True)
            else:
                st.info("No product column found in dataset to plot sentiment by product.")

        # Timeline chart if date column exists
        has_date = "date" in df.columns
        if has_date:
            st.markdown("---")
            st.markdown("##### 5. Sentiment Trends Over Time")
            try:
                t_df = df.copy()
                t_df["date"] = pd.to_datetime(t_df["date"])
                t_grouped = t_df.groupby([t_df["date"].dt.date, "predicted_sentiment"]).size().unstack(fill_value=0)
                st.line_chart(t_grouped, use_container_width=True)
            except Exception:
                st.info("Could not parse date column into time series.")


# ──────────────────────────────────────────────────────────────────────────────
# 7. EXPORT PAGE
# ──────────────────────────────────────────────────────────────────────────────
elif nav_selection == "Export":
    st.title("Data Export & Reporting")
    st.markdown("Download analyzed datasets, filtered results, and executive summaries.")

    df = st.session_state.get("analyzed_df")

    if df is None or len(df) == 0:
        st.warning("No analyzed dataset available for export.")
    else:
        st.markdown("### 📥 Download Options")

        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown("##### Full Analyzed Dataset")
            st.write(f"Contains all {len(df):,} records with predicted sentiments and probabilities.")
            full_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Analyzed CSV",
                data=full_csv,
                file_name=f"full_analyzed_sentiment_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with exp_col2:
            st.markdown("##### Executive Summary CSV")
            # Build aggregate summary by product or sentiment
            prod_col = "product_name" if "product_name" in df.columns else ("product" if "product" in df.columns else None)
            if prod_col:
                summary_df = df.groupby(prod_col).agg(
                    total_reviews=("predicted_sentiment", "count"),
                    avg_confidence=("confidence", "mean"),
                    positive_count=("predicted_sentiment", lambda s: (s == "positive").sum()),
                    neutral_count=("predicted_sentiment", lambda s: (s == "neutral").sum()),
                    negative_count=("predicted_sentiment", lambda s: (s == "negative").sum()),
                ).reset_index()
                summary_df["positive_pct"] = (summary_df["positive_count"] / summary_df["total_reviews"] * 100).round(2)
                summary_df["neutral_pct"] = (summary_df["neutral_count"] / summary_df["total_reviews"] * 100).round(2)
                summary_df["negative_pct"] = (summary_df["negative_count"] / summary_df["total_reviews"] * 100).round(2)
                summary_df["avg_confidence"] = (summary_df["avg_confidence"] * 100).round(2)
            else:
                summary_df = df.groupby("predicted_sentiment").agg(
                    count=("confidence", "count"),
                    avg_confidence=("confidence", "mean"),
                ).reset_index()

            st.write("Contains aggregated metrics and percentage distributions.")
            summary_csv = summary_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Summary CSV",
                data=summary_csv,
                file_name=f"sentiment_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("##### Executive Summary Preview")
        st.dataframe(summary_df, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# 8. MODEL INFORMATION PAGE
# ──────────────────────────────────────────────────────────────────────────────
elif nav_selection == "Model Information":
    st.title("Model Architecture & Technical Specification")
    st.markdown("Detailed breakdown of the production Phase 4 sentiment classification system.")

    st.markdown(
        """
        ### 🏆 Production Model Specification
        - **Model Name:** TF-IDF + 12 Linguistic Features + Logistic Regression
        - **Target Classes:** 3 Classes (`negative`, `neutral`, `positive`)
        - **Test Set Macro F1:** `0.6657` (Top Project Benchmark)
        - **Test Set Accuracy:** `66.69%`
        - **Training Dataset Size:** `102,076` clean sentences
        - **Status:** Phase 4 Benchmark Model (Production Deployed)
        """
    )

    st.markdown("---")
    st.markdown("### 🔬 Feature Engineering Pipeline")

    st.markdown(
        """
        The production pipeline combines high-dimensional sparse n-gram lexical representations with dense, interpretable psychological and syntactic features:
        
        1. **Text Preprocessing:**
           - Preserves punctuation, casing, emojis, and repetition for linguistic feature extraction.
           - Lowercasing and whitespace normalization for TF-IDF matrix generation.
           
        2. **TF-IDF Vectorization:**
           - Unigrams + Bigrams `(1, 2)`
           - Sublinear term frequency scaling: `log(1 + tf)`
           - Min document frequency: `min_df=2`, Max document frequency: `max_df=0.95`
           - Total vocabulary size: `123,493` features
           
        3. **12 Interpretable Linguistic Features:**
           - **Group A (Text Statistics):** `word_count`, `character_count`, `sentence_length`
           - **Group B (Syntax & Casing):** `exclamation_count`, `question_count`, `uppercase_ratio`, `uppercase_word_count`
           - **Group C (Sentiment Cues):** `negation_count` (20 markers), `intensifier_count` (11 degree adverbs), `contrast_word_count` (8 discourse markers)
           - **Group D (Visual & Repetition):** `emoji_count` (Unicode detection), `repeated_character_count` (3+ identical consecutive characters)
           
        4. **Feature Scaling & Fusion:**
           - Dense features standardized using `StandardScaler` fitted exclusively on training data.
           - Sparse concatenation via `scipy.sparse.hstack`.
           
        5. **Classification Engine:**
           - Multinomial Logistic Regression (`L-BFGS` solver, `C=1.0`).
           - Balanced class weighting to handle neutral class dominance.
        """
    )


# ──────────────────────────────────────────────────────────────────────────────
# 9. EXPERIMENTS RESULTS PAGE
# ──────────────────────────────────────────────────────────────────────────────
elif nav_selection == "Experiments":
    st.title("Experimental Research Comparison (Phases 2 – 7)")
    st.markdown("Consolidated benchmarks from the rigorous empirical evaluations across classical ML, sequential deep learning, and transformer architectures.")

    exp_data = [
        {"Phase": "Phase 2", "Architecture": "TF-IDF + Logistic Regression", "Training Samples": "102,076", "Accuracy": 0.6629, "Macro F1": 0.6611, "Notes": "Classical baseline"},
        {"Phase": "Phase 4", "Architecture": "TF-IDF + 12 Linguistic Features + Logistic Regression", "Training Samples": "102,076", "Accuracy": 0.6669, "Macro F1": 0.6657, "Notes": "🏆 Top Project Benchmark"},
        {"Phase": "Phase 5", "Architecture": "Bidirectional LSTM (BiLSTM from scratch)", "Training Samples": "102,076", "Accuracy": 0.6579, "Macro F1": 0.6557, "Notes": "Sequential deep learning"},
        {"Phase": "Phase 6", "Architecture": "Hybrid BiLSTM + Group B Features", "Training Samples": "102,076", "Accuracy": 0.6649, "Macro F1": 0.6611, "Notes": "Sequential + syntax hybrid"},
        {"Phase": "Phase 7", "Architecture": "Pretrained DistilBERT Transformer", "Training Samples": "10,000", "Accuracy": 0.5769, "Macro F1": 0.5697, "Notes": "⚠️ Constrained feasibility experiment (10k samples, 1 epoch, CPU-only)"},
    ]

    exp_df = pd.DataFrame(exp_data)

    st.markdown("### 📊 Benchmark Comparison Table")
    st.dataframe(
        exp_df.style.format({"Accuracy": "{:.2%}", "Macro F1": "{:.4f}"}),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### 📈 Macro F1 Performance Comparison")

    f1_chart_df = pd.DataFrame({
        "Model": [d["Architecture"] for d in exp_data],
        "Macro F1": [d["Macro F1"] for d in exp_data],
    }).set_index("Model")

    st.bar_chart(f1_chart_df, use_container_width=True)

    st.info(
        """
        **Scientific Integrity Note on Phase 7:**
        DistilBERT was evaluated under severe CPU computational constraints (10,000 stratified samples, 1 epoch). 
        Full dataset fine-tuning on CPU would require ~42 hours. Hence, Phase 4 remains the project's highest-performing benchmark across all completed experiments.
        """
    )


# ── Global Footer ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer-bar">AI-Based Sentiment Analysis System | Project Demo | Production Dashboard</div>',
    unsafe_allow_html=True,
)
