# ReguSense AI: Automated Regulatory Intelligence for the AI Era

> *Turning the federal "noise" of AI policy into a structured, searchable knowledge base for compliance teams.*


## 💡 The Problem: "The AI Policy Firehose"

As of 2026, AI regulations are moving faster than companies can read them. Searching the **Federal Register** for "Artificial Intelligence" typically returns 200+ documents a week, but **90% are noise** (mentions of AI in healthcare billing, military recruitment, or climate modeling).

Compliance officers don't need *more* data; they need **high-fidelity signals.**

## 🧠 The Solution

**ReguSense AI** is an end-to-end data pipeline that ingests, filters, and vectorizes federal regulatory notices. It uses a **multi-stage filtration system** to ensure only primary AI policy enters the knowledge base, reducing data clutter by over 90% while maintaining 100% relevance.

---

## 🏗️ System Architecture

1. **Ingestion:** A daily Airflow DAG fetches significant notices from the Federal Register API.
2. **The "AI Judge":** A deterministic LLM agent (GPT-4o-mini) analyzes abstracts using **logprobs** to score "Semantic Centricity." Documents with low relevance or confidence scores are discarded.
3. **Warehouse:** Cleaned data and metadata are stored in **MotherDuck**, providing a serverless, cloud-ready SQL layer.
4. **Vector Knowledge Base:** Filtered documents are embedded via OpenAI and upserted into **Pinecone** for semantic retrieval.
5. **Interface:** A Streamlit-based "Regulatory Assistant" allows users to query the law in natural language (e.g., *"What are my 2026 audit requirements for high-risk medical AI?"*).

---

## 🛠️ The Tech Stack

* **Orchestration:** Apache Airflow (Dockerized)
* **Data Warehouse:** MotherDuck (Cloud DuckDB)
* **Transformation:** dbt-core
* **Vector DB:** Pinecone
* **Models:** OpenAI `gpt-4o-mini` (Filtering) & `text-embedding-3-small` (Embeddings)
* **Frontend:** Streamlit

---

## 🤝 Contact & Portfolio

**Author:** Shubhanshu Vaibhav


