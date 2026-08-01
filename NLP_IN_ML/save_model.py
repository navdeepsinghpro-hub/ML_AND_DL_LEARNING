import string
from pathlib import Path

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

MODEL_PATH = Path(__file__).parent / "model_lr.pkl"
VECTORIZER_PATH = Path(__file__).parent / "tf_vectorizer.pkl"
LABEL_MAP_PATH = Path(__file__).parent / "label_map.pkl"

nltk.download('punkt')
nltk.download('stopwords')


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join(ch for ch in text if not ch.isdigit())
    text = ''.join(ch for ch in text if ch.isascii())

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return ' '.join([token for token in tokens if token not in stop_words])


def main() -> None:
    df = pd.read_csv(Path(__file__).parent / 'train.txt', sep=';', header=None, names=['text', 'emotion'])
    df['text'] = df['text'].astype(str).apply(preprocess_text)

    unique_emotions = df['emotion'].unique().tolist()
    emotion_numbers = {emo: idx for idx, emo in enumerate(unique_emotions)}
    df['emotion'] = df['emotion'].map(emotion_numbers)

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['emotion'], test_size=0.20, random_state=42)

    vectorizer = CountVectorizer()
    X_train_tf = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tf, y_train)

    label_map = {idx: emo for emo, idx in emotion_numbers.items()}

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(label_map, LABEL_MAP_PATH)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved label map to {LABEL_MAP_PATH}")


if __name__ == '__main__':
    main()
