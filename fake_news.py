import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. LOAD DATASETS
# ============================================================

TRAIN_FILE = r"C:\Users\Zuni\Desktop\fake_news_system\dataset\train.csv"
TEST_FILE = r"C:\Users\Zuni\Desktop\fake_news_system\dataset\test.csv"

print("=" * 60)
print("              FAKE NEWS DETECTION SYSTEM")
print("=" * 60)

print("\nLoading datasets...")

train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)

print("Training dataset:", train_data.shape)
print("Testing dataset :", test_data.shape)


# ============================================================
# 2. CHECK COLUMNS
# ============================================================

required_train_columns = ["title", "text", "class"]
required_test_columns = ["title", "text"]

for column in required_train_columns:

    if column not in train_data.columns:
        raise ValueError(
            f"Column '{column}' is missing from train.csv"
        )

for column in required_test_columns:

    if column not in test_data.columns:
        raise ValueError(
            f"Column '{column}' is missing from test.csv"
        )

print("\nRequired columns found successfully.")


# ============================================================
# 3. CLEAN TEXT FUNCTION
# ============================================================

def clean_text(text):

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

print("\nPreparing text data...")

train_data["title"] = train_data["title"].fillna("")
train_data["text"] = train_data["text"].fillna("")

test_data["title"] = test_data["title"].fillna("")
test_data["text"] = test_data["text"].fillna("")


# ============================================================
# 5. COMBINE TITLE + TEXT
# ============================================================

train_data["combined_text"] = (
    train_data["title"]
    + " "
    + train_data["text"]
)

test_data["combined_text"] = (
    test_data["title"]
    + " "
    + test_data["text"]
)


# Clean the combined text

train_data["combined_text"] = (
    train_data["combined_text"]
    .apply(clean_text)
)

test_data["combined_text"] = (
    test_data["combined_text"]
    .apply(clean_text)
)


# ============================================================
# 6. REMOVE EMPTY ARTICLES
# ============================================================

train_data = train_data[
    train_data["combined_text"].str.strip() != ""
]

test_data = test_data[
    test_data["combined_text"].str.strip() != ""
]


# ============================================================
# 7. CHECK CLASS LABELS
# ============================================================

# ============================================================
# 7. CLEAN CLASS LABELS
# ============================================================

train_data["class"] = (
    train_data["class"]
    .astype(str)
    .str.strip()
)

# Keep only valid labels
train_data = train_data[
    train_data["class"].isin(["Fake", "Real"])
].copy()


# ============================================================
# 8. PREPARE DATA
# ============================================================

X = train_data["combined_text"]
y = train_data["class"]

X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 9. TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    min_df=2,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_validation_tfidf = vectorizer.transform(X_validation)


# ============================================================
# 10. TRAIN MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    C=2.0
)

model.fit(
    X_train_tfidf,
    y_train
)

print("\nModel classes:", model.claases_)

# ============================================================
# 11. CALCULATE ACCURACY
# ============================================================

validation_predictions = model.predict(
    X_validation_tfidf
)

accuracy = accuracy_score(
    y_validation,
    validation_predictions
)


# ============================================================
# 12. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "fake_news_model.pkl"
)

joblib.dump(
    vectorizer,
    "tfidf_vectorizer.pkl"
)


# ============================================================
# 13. NEWS PREDICTION
# ============================================================

def predict_news(title, article):

    combined_text = (
        str(title) + " " + str(article)
    )

    cleaned_text = clean_text(
        combined_text
    )

    transformed_text = vectorizer.transform(
        [cleaned_text]
    )

    prediction = model.predict(
        transformed_text
    )[0]

    probabilities = model.predict_proba(
        transformed_text
    )[0]

    confidence = max(probabilities) * 100

    return prediction, confidence


# ============================================================
# 14. SIMPLE NEWS CHECKER
# ============================================================

print("\n" + "=" * 50)
print("       FAKE NEWS DETECTION SYSTEM")
print("=" * 50)

print(
    f"\nModel Accuracy: {accuracy * 100:.2f}%"
)

while True:

    title = input(
        "\nEnter news title (or type exit): "
    )

    if title.lower() == "exit":
        break

    article = input(
        "Enter news article: "
    )

    if article.lower() == "exit":
        break

    if not article.strip():

        print("\nPlease enter a news article.")

        continue


    prediction, confidence = predict_news(
        title,
        article
    )


    print("\n" + "=" * 50)
    print("                  RESULT")
    print("=" * 50)


    if prediction == "Real":

        print("\n✅ THIS NEWS IS REAL")

    elif prediction == "Fake":

        print("\n❌ THIS NEWS IS FAKE")

    else:

        print(
            f"\nPrediction: {prediction}"
        )


    print(
        f"\nModel Accuracy: {accuracy * 100:.2f}%"
    )

    print("=" * 50)


print("\nProgram closed.")