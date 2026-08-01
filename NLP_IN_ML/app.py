import string
from pathlib import Path

import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="🤖",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(to right,#0f172a,#1e293b,#334155);
}

.main-title{
    font-size:55px;
    font-weight:700;
    color:white;
    text-align:center;
}

.subtitle{
    text-align:center;
    color:#cbd5e1;
    font-size:20px;
    margin-bottom:40px;
}

.card{
    background:#1e293b;
    padding:25px;
    border-radius:18px;
    box-shadow:0px 0px 15px rgba(0,0,0,.4);
    text-align:center;
}

.metric{
    font-size:35px;
    font-weight:bold;
    color:#38bdf8;
}

.metric-title{
    color:white;
    font-size:18px;
}

.result{
    background:#22c55e;
    padding:20px;
    border-radius:15px;
    color:white;
    font-size:35px;
    text-align:center;
    font-weight:bold;
}

textarea{
    font-size:18px !important;
}

.stButton>button{
    width:100%;
    background:#2563eb;
    color:white;
    border-radius:12px;
    height:55px;
    font-size:22px;
    font-weight:bold;
    border:none;
}

.stButton>button:hover{
    background:#1d4ed8;
}

</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------

st.markdown(
    "<div class='main-title'>🤖 AI Emotion Detection</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Detect Human Emotions using NLP & Logistic Regression</div>",
    unsafe_allow_html=True
)

MODEL_PATH = Path(__file__).parent / "model_lr.pkl"
VECTORIZER_PATH = Path(__file__).parent / "tf_vectorizer.pkl"
LABEL_MAP_PATH = Path(__file__).parent / "label_map.pkl"

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


@st.cache_data(show_spinner=False)
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_map = joblib.load(LABEL_MAP_PATH)
    return model, vectorizer, label_map


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join(ch for ch in text if not ch.isdigit())
    text = ''.join(ch for ch in text if ch.isascii())

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return ' '.join([token for token in tokens if token not in stop_words])


model, vectorizer, label_map = load_model()

# -------------------- INFO CARDS --------------------

c1,c2,c3,c4=st.columns(4)

with c1:
    st.markdown("""
    <div class="card">
    <div class="metric">86%</div>
    <div class="metric-title">Accuracy</div>
    </div>
    """,unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
    <div class="metric">6</div>
    <div class="metric-title">Emotions</div>
    </div>
    """,unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
    <div class="metric">ML</div>
    <div class="metric-title">Logistic Regression</div>
    </div>
    """,unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="card">
    <div class="metric">NLP</div>
    <div class="metric-title">CountVectorizer</div>
    </div>
    """,unsafe_allow_html=True)

st.write("")

# -------------------- INPUT --------------------

st.subheader("📝 Enter Text")

text = st.text_area(
    "",
    height=180,
    placeholder="Example: I am feeling very happy today!"
)

# -------------------- BUTTON --------------------

predict = st.button("🔍 Predict Emotion")

# -------------------- OUTPUT --------------------

if predict:
    if not text:
        st.warning("Please enter some text before predicting.")
    else:
        cleaned_text = preprocess_text(text)
        vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]

        emotion_label = label_map.get(int(prediction), "Unknown")
        confidence = round(float(probabilities[int(prediction)]) * 100, 2)

        st.markdown(
            f"""
            <div class="result">
            {emotion_label.title()} ({confidence}%)
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")
        st.progress(int(confidence))
        st.success(f"Confidence : {confidence}%")

        st.write("")
        st.subheader("📊 Prediction Probability")

        sorted_probs = sorted(
            [(label_map[idx], prob) for idx, prob in enumerate(probabilities)],
            key=lambda x: x[1],
            reverse=True,
        )

        for label, prob in sorted_probs:
            pct = round(float(prob) * 100, 2)
            st.write(f"{label.title()} : {pct}%")
            st.progress(int(pct))

        st.write("")
        st.info("💡 Keep smiling! Positive emotions make life beautiful. ✨")

# -------------------- FOOTER --------------------

st.write("---")
st.caption("Developed by Navdeep Singh | B.Tech AIML")