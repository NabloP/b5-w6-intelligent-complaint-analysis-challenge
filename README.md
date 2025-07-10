
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

---

## 📁 Project Structure

<!-- TREE START -->
📁 Project Structure

b5-w6-intelligent-complaint-analysis-challenge/
├── LICENSE
├── README.md
├── requirements.txt
├── app/
│   ├── app.py
│   ├── ui_helpers.py
├── data/
│   ├── interim/
│   │   ├── filtered_complaints.csv
│   ├── plots/
│   │   ├── complaint_volume_by_product.png
│   │   ├── distribution_complaint_narrative_lengths.png
│   │   ├── missing_values_heatmap.png
│   │   ├── monthly_complaint_volume_over_time.png
│   ├── processed/
│   └── raw/
│       ├── complaints.csv
├── notebooks/
│   ├── task-1-eda-preprocessing.ipynb
│   ├── task-2-embedding-indexing.ipynb
│   ├── task-3-rag.ipynb
├── scripts/
│   ├── embedding_runner.py
│   ├── generate_tree.py
│   ├── rag_pipeline.py
│   ├── run_streamlit.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── chunking/
│   │   ├── embedding_generator.py
│   │   ├── text_chunker.py
│   │   ├── text_cleaner.py
│   │   ├── vector_store_builder.py
│   ├── eda/
│   │   ├── eda_visualizer.py
│   │   ├── schema_auditor.py
│   └── rag/
│       ├── answer_generator.py
│       ├── chroma_loader.py
│       ├── prompt_template.py
│       ├── qualitative_evaluator.py
│       ├── retriever.py
├── ui/
│   ├── app.py
└── vector_store/
    ├── chroma.sqlite3
    └── chroma_db/
        ├── chroma.sqlite3
        └── 8783f26e-2984-4f72-b4b2-dbd817307c15/
            ├── data_level0.bin
            ├── header.bin
            ├── index_metadata.pickle
            ├── length.bin
            ├── link_lists.bin
<!-- TREE END -->
---

## ✅ Status (as of July 8)

| Decision                          | Rationale                                                                                                  |
|----------------------------------|------------------------------------------------------------------------------------------------------------|
| ✅ **ChromaDB** over FAISS         | Better metadata handling, easier integration with LangChain for future RAG deployment                       |
| ✅ **all-MiniLM-L6-v2**            | Optimal trade-off between **semantic precision** and **computational efficiency**—ideal for CrediTrust’s real-time requirements |
| ✅ **Recursive Chunking**          | Ensures **no context loss** for long-form complaints while enabling short narratives to pass unaltered      |
| ✅ **Modular RAG Retriever & Generator** | Enables flexible, scalable retrieval and generation pipeline with streaming and dynamic memory            |
| ✅ **Google Gemini LLM Integration**     | Leverages state-of-the-art generative models for context-aware, evidence-backed answers                    |
| ✅ **Streamlit Interactive Chatbot**    | Provides a minimal, fast, and user-friendly interface with live streaming, theme toggling, and source transparency |


---

## 📊 Task Progress Tracker

| Task # | Task Name                          | Status       | Description                                                                     |
|--------|------------------------------------|--------------|---------------------------------------------------------------------------------|
| 1      | Exploratory Data Analysis (EDA)    | ✅ Completed | Visualized complaint volumes, lengths, nulls; filtered data for target products.|
| 2      | Text Chunking & Embedding          | ✅ Completed | Applied RecursiveCharacterTextSplitter, generated embeddings, stored with FAISS.|
| 3      | RAG Pipeline Core Logic            | ✅ Completed | Building retrieval + generation logic and qualitative evaluation.               |
| 4      | Interactive Streamlit Interface    | ✅ Completed | Streamlit chatbot with source transparency and real-time querying.              |

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