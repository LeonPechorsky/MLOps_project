import os
import time
import requests
import streamlit as st

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Text Sentiment Analysis",
    layout="centered"
)

# -------------------- CUSTOM STYLES --------------------
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2E86AB;
            text-align: center;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .prediction-box {
            padding: 1.5rem;
            border-radius: 10px;
            background-color: #f8f9fa;
            border-left: 6px solid #2E86AB;
            margin: 1.5rem 0;
        }
        .positive {
            color: #28a745;
        }
        .negative {
            color: #dc3545;
        }
        .neutral {
            color: #ffc107;
        }
        .footer {
            margin-top: 3rem;
            font-size: 0.8rem;
            color: #aaa;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- UI HEADER --------------------
st.markdown('<div class="main-header">Text Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by ClearML Serving · Model from Registry · No local ML</div>', unsafe_allow_html=True)

# Sidebar for endpoint
with st.sidebar:
    st.header("⚙️ Configuration")
    endpoint = st.text_input(
        "Serving Endpoint URL",
        value=os.environ.get("SERVING_URL", "http://localhost:9090/serve/sentiment_analyze")
    )
    st.caption("Expects JSON: `{\"text\": \"...\"}`")

# Main input area
text = st.text_area(
    "📝 Enter your text",
    height=120,
    placeholder="e.g. This product is absolutely amazing!",
    label_visibility="collapsed"
)

# Center the button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_clicked = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

# -------------------- PREDICTION LOGIC --------------------
if predict_clicked:
    if not text or not text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Calling inference endpoint..."):
            try:
                start_time = time.perf_counter()
                response = requests.post(endpoint, json={"text": text}, timeout=10)
                latency_ms = (time.perf_counter() - start_time) * 1000
                response.raise_for_status()
                data = response.json()

                # Parse prediction
                pred = data["predictions"][0]
                label = pred["label"]       # "1" or "0"
                score = pred.get("score", 0.0)

                # Map label
                if label == "1":
                    display_label = "Positive"
                    sentiment_class = "positive"
                elif label == "0":
                    display_label = "Negative"
                    sentiment_class = "negative"
                else:
                    display_label = label
                    sentiment_class = "neutral"

                # ----- Result Display -----
                st.markdown(f'<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f"### 🎯 Prediction: **{display_label}**")
                if sentiment_class == "positive":
                    st.markdown(f'<span class="positive" style="font-size:1.2rem;">😃 Positive sentiment</span>', unsafe_allow_html=True)
                elif sentiment_class == "negative":
                    st.markdown(f'<span class="negative" style="font-size:1.2rem;">😞 Negative sentiment</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="neutral" style="font-size:1.2rem;">😐 Neutral / Unknown</span>', unsafe_allow_html=True)

                # Metrics
                col1, col2 = st.columns(2)
                col1.metric("Confidence", f"{score:.1%}")
                col2.metric("Latency", f"{latency_ms:.0f} ms")

                # Raw response expander (optional)
                with st.expander("📄 Show raw response"):
                    st.json(data)

                st.markdown('</div>', unsafe_allow_html=True)

            except requests.exceptions.ConnectionError:
                st.error(f"❌ Cannot connect to endpoint: `{endpoint}`\n\nMake sure the serving service is running.")
            except requests.exceptions.Timeout:
                st.error("⏰ Request timed out (10 seconds). The endpoint might be overloaded.")
            except requests.exceptions.HTTPError as e:
                st.error(f"⚠️ Server error {e.response.status_code}: {e.response.text[:200]}")
            except (KeyError, ValueError) as e:
                st.error(f"❌ Unexpected response format: {e}")

# Footer
st.markdown('<div class="footer">Built with Streamlit · Model served by ClearML · All inference happens on the server</div>', unsafe_allow_html=True)