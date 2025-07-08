
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

## 🏗 Project Components Completed (Tasks 1 & 2)

### ✅ Clean Ingestion & Exploratory Data Analysis (Task 1)

We ingested and analyzed over **650,000 consumer complaints** from the **Consumer Financial Protection Bureau (CFPB)**, performing:
- 📊 **Complaint Volume Analysis** by product, time, and narrative length
- 🧹 **Schema Audit & Missingness Diagnostics** to ensure only business-relevant, high-integrity fields were retained
- 📝 **Text Preprocessing** using a **lossless cleaning pipeline** (`src/chunking/text_cleaner.py`) to preserve linguistic nuances crucial for semantic search

Key Output:  
`data/interim/filtered_complaints.csv`  
Containing ~270,000 cleaned and filtered complaint narratives across **five strategic financial products**:  
Credit Cards, Personal Loans, Buy Now Pay Later (BNPL), Savings Accounts, and Money Transfers.

---

### ✅ Text Chunking, Embedding & ChromaDB Indexing (Task 2)

To prepare complaint narratives for efficient semantic search, we built a **modular chunking and embedding pipeline** using:

| Component | Implementation Details |
|-----------|------------------------|
| **Chunking Logic** | Used **LangChain’s RecursiveCharacterTextSplitter** (`src/chunking/text_chunker.py`) with:<br> • Chunk Size: **500 tokens** <br> • Overlap: **50 tokens** <br> This preserves context while staying within embedding model limits. |
| **Embedding Model** | Chose **all-MiniLM-L6-v2** for:<br> • ⚡ **Speed**: Fast inference on CPU<br> • 🎯 **Accuracy**: Strong semantic matching performance in general language and complaint-style text |
| **Vector Store** | Created a **ChromaDB** vector store (`src/chunking/vector_store_builder.py`), storing:<br> • Embeddings<br> • Associated metadata (Product, Date, Complaint ID, Raw Text)<br> • Persisted under `/vector_store/` for fast, reusable semantic retrieval |

We opted for **ChromaDB** over FAISS to take advantage of:
- **Native metadata storage**
- **Lightweight integration with LangChain**
- **Ease of deployment in production workflows**

Key Script:  
`scripts/embedding_runner.py`  
Allows end-to-end execution of chunking, embedding, and vector index creation in a single command.

---

## 🔗 Key Technology Decisions & Justifications

| Decision | Rationale |
|----------|-----------|
| ✅ **ChromaDB** over FAISS | Better metadata handling, easier integration with LangChain for future RAG deployment |
| ✅ **all-MiniLM-L6-v2** | Optimal trade-off between **semantic precision** and **computational efficiency**—ideal for CrediTrust’s real-time requirements |
| ✅ **Recursive Chunking** | Ensures **no context loss** for long-form complaints while enabling short narratives to pass unaltered |

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
│   │   ├── filtered_complaints.csv
│   ├── processed/
│   └── raw/
│       ├── complaints.csv
├── notebooks/
│   ├── task-1-eda-preprocessing.ipynb
│   ├── task-2-embedding-indexing.ipynb
├── scripts/
│   ├── embedding_runner.py
│   ├── generate_tree.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   ├── chunking/
│   │   ├── embedding_generator.py
│   │   ├── text_chunker.py
│   │   ├── text_cleaner.py
│   │   ├── vector_store_builder.py
│   └── eda/
│       ├── eda_visualizer.py
│       ├── schema_auditor.py
└── vector_store/
<!-- TREE END -->
---

## ✅ Interim Status (as of July 6)

| Task # | Task Description | Status | Key Deliverables |
|--------|------------------|--------|------------------|
| **1** | Data Exploration & Cleaning | ✅ Completed | Cleaned dataset, EDA notebooks, schema diagnostics |
| **2** | Chunking & Embedding with ChromaDB | ✅ Completed | Modular scripts, persisted vector store |
| **3** | RAG Pipeline & Evaluation | 🔵 In Progress | Retriever + Generator logic under development |
| **4** | Interactive Streamlit Chatbot | 🔵 Pending | Planned for Task 4 |

---

## 📊 Task Progress Tracker

| Task # | Task Name                         | Status      | Description |
|--------|------------------------------------|-------------|-------------|
| 1      | Exploratory Data Analysis (EDA)    | ✅ Completed | Visualized complaint volumes, lengths, nulls; filtered data for target products. |
| 2      | Text Chunking & Embedding          | ✅ Completed | Applied RecursiveCharacterTextSplitter, generated embeddings, stored with FAISS. |
| 3      | RAG Pipeline Core Logic            | 🔵 In Progress | Building retrieval + generation logic and qualitative evaluation. |
| 4      | Interactive Streamlit Interface    | 🔵 Pending | Streamlit chatbot with source transparency and real-time querying. |

---

## 🔍 Next Steps

1. Build the **Retriever + LLM Prompt** pipeline using precomputed embeddings.
2. Conduct **qualitative evaluation** using a curated question bank.
3. Deploy an **interactive Streamlit app** for CrediTrust’s internal teams.
4. Implement **explainability layers** to enhance trust and regulatory compliance.

---

## 🚀 Planned Final Deliverables

| Deliverable | Format / Location |
|------------|-------------------|
| Cleaned Complaint Dataset | `data/interim/filtered_complaints.csv` |
| EDA & Visuals | `notebooks/task-1-eda-preprocessing.ipynb` |
| Embedding & Indexing Pipeline | `scripts/embedding_runner.py` |
| ChromaDB Vector Store | `/vector_store/` |
| RAG Pipeline | `src/rag_pipeline.py` |
| Streamlit Chatbot | `app/app.py` |

---

## 📚 References

- LangChain Documentation
- ChromaDB
- Hugging Face Sentence Transformers
- Streamlit
- CFPB Open Data

---

## 👤 Author

**Nabil Mohamed**  
10 Academy AIM Bootcamp Participant  
GitHub: [@NabloP](https://github.com/NabloP)