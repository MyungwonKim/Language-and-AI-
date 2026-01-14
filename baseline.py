import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

FILE_PATH = "data/your_dataset.csv" # Change this to the preprocessed dataset path
TARGET_COL = 'extrovert'          
USER_ID_COL = 'auhtor_ID'
TEXT_COL = 'clean_text' # Change if your text column has a different name


def run_baseline():
    # Load Data
    print(f"--- Loading {FILE_PATH} ---")
    df = pd.read_csv(FILE_PATH)
    df = df.dropna(subset=[TEXT_COL, TARGET_COL, USER_ID_COL])
    
    # Train test split
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df[USER_ID_COL]))

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    X_train = train_df[TEXT_COL]
    y_train = train_df[TARGET_COL]
    X_test = test_df[TEXT_COL]
    y_test = test_df[TARGET_COL]

    print(f"Train Size: {len(X_train)} | Test Size: {len(X_test)}")
    print("-" * 30)

    # TF-IDF + Logistic Regression
    print("running Baseline: TF-IDF + LogReg...")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    lr_preds = pipeline.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, lr_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, lr_preds, average='binary')

    print(">> Results for TF-IDF + LogReg:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nDetailed Report:\n", classification_report(y_test, lr_preds))
    print("-" * 30)

if __name__ == "__main__":
    run_baseline()
