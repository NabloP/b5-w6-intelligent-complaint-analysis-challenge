
# B5W6: Intelligent Complaint Analysis — Week 6 Challenge | 10 Academy

## 🗂 Challenge Context

This repository documents the submission for 10 Academy’s **B5W6: Intelligent Complaint Analysis** challenge.

CrediTrust Financial, a fast-growing digital finance company operating across East Africa, faces challenges in identifying, understanding, and acting on large volumes of unstructured customer complaints. This project builds an AI-driven Retrieval-Augmented Generation (RAG) pipeline to:

- Transform unstructured complaint narratives into actionable business insights
- Empower non-technical teams to ask questions and receive grounded, evidence-based answers
- Shift the organization from reactive to proactive customer issue resolution

This project includes:
- 🧹 Clean ingestion and processing of real-world financial complaints data
- 📊 Exploratory Data Analysis (EDA) of complaint volumes, narratives, and product distributions
- 🔍 Text chunking and semantic embedding for efficient vector search
- 🧠 RAG pipeline combining FAISS-based retrieval with LLM summarization
- 🌐 Interactive Streamlit chatbot for internal teams (Product, Compliance, Support)

---

## 🔧 Project Setup

1. Clone the repository:

```bash
git clone https://github.com/NabloP/b5-w6-intelligent-complaint-analysis-challenge.git
cd b5-w6-intelligent-complaint-analysis-challenge
```

2. Create and activate the virtual environment:

**On Windows:**
```bash
python -m venv complaint-analysis-challenge
.\complaint-analysis-challenge\Scripts\Activate.ps1
```

**On macOS/Linux:**
```bash
python3 -m venv complaint-analysis-challenge
source complaint-analysis-challenge/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ⚙️ CI/CD (GitHub Actions)

This project uses GitHub Actions for Continuous Integration. On every `push` or `pull_request`, the following checks run:

- Repository checkout
- Python 3.10 setup
- Install and validate dependencies

CI workflow is defined in:

    .github/workflows/unittests.yml

---

## 🔐 Complaint Analysis Business Understanding

### 1. The Need for Automated Complaint Analysis

With thousands of complaints per month across five key financial products (Credit Cards, Personal Loans, BNPL, Savings, Transfers), internal teams at CrediTrust struggle with:
- Slow manual complaint review
- Lack of systematic insights
- Reactive risk management

An AI-powered complaint analysis system enables:
- Faster identification of top complaint themes
- Early warning for compliance breaches or fraud
- Evidence-backed decision making

### 2. RAG for Strategic Insights

RAG systems combine the power of:
- **Retrieval**: Using vector search (FAISS) to find the most relevant complaint narratives
- **Augmentation**: Feeding retrieved context into a Large Language Model
- **Generation**: Producing a concise, context-grounded answer

This architecture ensures answers are both insightful and verifiable.

---

<!-- TREE START -->
📁 Project Structure

solar-challenge-week1/
├── LICENSE
├── README.md
├── requirements.txt
├── app/
│   ├── app.py
│   ├── ui_helpers.py
├── data/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── notebooks/
│   ├── task-1-eda-preprocessing.ipynb
│   ├── task-2-embedding-indexing.ipynb
├── scripts/
│   ├── generate_tree.py
├── src/
│   ├── __init__.py
│   ├── chunking_module.py
│   ├── data_loader.py
│   ├── embedding_module.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   ├── text_cleaner.py
└── vector_store/
<!-- TREE END -->
---

## ✅ Interim Status (as of July 6)

- ✅ **Task 1 complete:** Data exploration and cleaning completed; interim dataset saved.
- ✅ **Task 2 complete:** Text chunking, embedding with `all-MiniLM-L6-v2`, FAISS vector store created.
- 🔵 Task 3 (RAG pipeline & evaluation): In progress
- 🔵 Task 4 (Streamlit chatbot): Pending

---

## 📊 Task Progress Tracker

| Task # | Task Name                         | Status      | Description |
|--------|------------------------------------|-------------|-------------|
| 1      | Exploratory Data Analysis (EDA)    | ✅ Completed | Visualized complaint volumes, lengths, nulls; filtered data for target products. |
| 2      | Text Chunking & Embedding          | ✅ Completed | Applied RecursiveCharacterTextSplitter, generated embeddings, stored with FAISS. |
| 3      | RAG Pipeline Core Logic            | 🔵 In Progress | Building retrieval + generation logic and qualitative evaluation. |
| 4      | Interactive Streamlit Interface    | 🔵 Pending | Streamlit chatbot with source transparency and real-time querying. |

---

## 🚀 Planned Deliverables

| Deliverable                              | Format                           |
|------------------------------------------|-----------------------------------|
| Cleaned complaint dataset                | `data/interim/filtered_complaints.csv` |
| EDA Notebook                             | `notebooks/task_1_eda_preprocessing.ipynb` |
| Embedding & Indexing Module              | `src/embedding_module.py`         |
| Persisted Vector Store                   | `vector_store/`                   |
| RAG Pipeline Module                      | `src/rag_pipeline.py`             |
| Interactive Chatbot                      | `app/app.py`                      |


---

## 🔗 References

Key sources used for this challenge:

- [LangChain Documentation](https://python.langchain.com/)
- [FAISS by Facebook AI](https://github.com/facebookresearch/faiss)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Consumer Financial Protection Bureau (CFPB)](https://www.consumerfinance.gov/data-research/consumer-complaints/)

---

## 👤 Author

**Nabil Mohamed**  
AIM Bootcamp Participant  
GitHub: [@NabloP](https://github.com/NabloP)
