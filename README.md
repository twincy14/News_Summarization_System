# 📰 AI News Summarization & ROUGE Benchmarking Engine

An end-to-end NLP system for abstractive news article summarization and automated ROUGE evaluation. Powered by HuggingFace Transformer models (`facebook/bart-large-cnn`) and built with an interactive Streamlit web dashboard.

---

## 🌟 Key Features

- **Abstractive News Summarization**: Generates natural, human-like summaries using state-of-the-art Sequence-to-Sequence BART models (`facebook/bart-large-cnn`).
- **Flexible Summary Output Formats**:
  - 📌 **Executive Key Takeaways (Structured)**: Bulleted thematic breakdowns (The Milestone, Key Drivers, Challenges, Looking Ahead).
  - 📝 **Standard Abstractive Paragraph**: Concise paragraph format.
- **Real-Time ROUGE Benchmarking**: Computes **ROUGE-1** (unigram), **ROUGE-2** (bigram), and **ROUGE-L** (longest common subsequence) scores with Precision, Recall, and F1 metrics.
- **Dataset Integration**: Pre-loaded with HuggingFace `CNN/DailyMail (v3.0.0)` test split for benchmarking sample articles.
- **Interactive Web Interface**: Streamlit dashboard featuring custom text inputs, live metric cards, and dataset sample explorer.

---

## 📁 Repository Structure

```text
News_Summarization_System/
├── app.py              # Streamlit Web Application & Interactive UI
├── dataset.py          # Module 1: HuggingFace CNN/DailyMail Data Collection
├── preprocess.py       # Module 2: Text Cleaning & Noise Stripping
├── summarizer.py       # Module 3: BART Abstractive Summarization Engine
├── evaluate.py         # Module 4: ROUGE Evaluation Metric Engine
├── main.py             # Module 5: Automated CLI Pipeline & Benchmark Runner
├── requirements.txt    # Project Python Dependencies
└── README.md           # Project Documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/twincy14/News_Summarization_System.git
cd News_Summarization_System
```

### 2. Create Virtual Environment & Install Dependencies
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## 🚀 Running the Application

### Option A: Launch Streamlit Web UI (Recommended)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to use the interactive dashboard.

### Option B: Run Automated Pipeline (CLI)
```bash
python main.py
```
This executes the end-to-end pipeline on sample dataset articles and displays per-article ROUGE scores in the terminal.

---

## 📊 ROUGE Metrics Explained

| Metric | Focus | Description |
| :--- | :--- | :--- |
| **ROUGE-1** | Unigram Overlap | Evaluates individual keyword and word-level overlap. |
| **ROUGE-2** | Bigram Overlap | Evaluates 2-word phrase matches and local fluency. |
| **ROUGE-L** | Longest Common Subsequence | Measures sentence structure and word order preservation. |

---

## 💻 Tech Stack

- **Model Engine**: HuggingFace `transformers` (`facebook/bart-large-cnn`), `torch`
- **Dataset**: HuggingFace `datasets` (`abisee/cnn_dailymail`)
- **Metrics**: `rouge-score`
- **Web App UI**: `streamlit`, `pandas`

---

## 👤 Author

- **GitHub**: [@twincy14](https://github.com/twincy14)
