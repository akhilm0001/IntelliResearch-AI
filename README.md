# IntelliResearch AI – Multi-Agent Research Assistant

An AI-powered research assistant that combines **Agentic AI, Retrieval-Augmented Generation (RAG), and web search** to automate research and generate structured research reports from PDF documents and online sources.

Built using CrewAI, LangChain, Ollama (Gemma 4), FAISS, Tavily, and Streamlit.

## 🚀 Project Overview

IntelliResearch AI is designed to simplify research by bringing together multiple AI agents that collaborate to collect, retrieve, analyze, and summarize information.

The system uses a locally running Large Language Model (LLM) through Ollama, allowing research tasks to be processed without depending on a paid cloud-based LLM API.

## ✨ Key Features

- **Multi-Agent Architecture:** Uses CrewAI to coordinate specialized research agents.
- **PDF Document Research:** Extracts and processes information from PDF documents.
- **Retrieval-Augmented Generation (RAG):** Retrieves relevant document context using FAISS.
- **Semantic Search:** Uses HuggingFace sentence embeddings for document retrieval.
- **Web Research:** Retrieves relevant online information using Tavily Search.
- **AI-Powered Analysis:** Compares and analyzes information gathered from different sources.
- **Automated Report Generation:** Produces structured research reports.
- **Local LLM Integration:** Uses Gemma 4 through Ollama for local inference.
- **Streamlit Interface:** Provides a user-friendly interface for interacting with the research system.

## 🧠 System Architecture

The project follows a multi-agent research workflow:

1. **Document Research Agent**
   - Retrieves relevant information from PDF documents.
   - Uses the RAG pipeline and FAISS vector database.

2. **Web Research Agent**
   - Searches for relevant information using Tavily.
   - Collects external research sources.

3. **Analyst Agent**
   - Evaluates and combines findings from document and web research.
   - Identifies important insights.

4. **Report Writer Agent**
   - Organizes the research findings.
   - Generates the final research report.

### Workflow

```text
              User Research Question
                       |
                       v
                CrewAI Manager
                       |
          +------------+------------+
          |                         |
          v                         v
   Document Research           Web Research
       Agent                      Agent
          |                         |
          v                         v
    PDF Documents               Tavily Search
          |                         |
          v                         v
    FAISS Retrieval            Web Results
          |                         |
          +------------+------------+
                       |
                       v
                  Analyst Agent
                       |
                       v
                Report Writer Agent
                       |
                       v
                Final Research Report
```

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| CrewAI | Multi-agent orchestration |
| Ollama | Local LLM execution |
| Gemma 4 | Language model |
| LangChain | RAG pipeline and integrations |
| FAISS | Vector database and similarity search |
| HuggingFace | Text embedding model |
| Sentence Transformers | Semantic embeddings |
| Tavily | Web search |
| PyPDF | PDF document loading |
| Streamlit | User interface |

## 📂 Project Structure

```text
IntelliResearch-AI/
│
├── app.py
├── project.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
├── IntelliResearch_Report.txt
│
└── Documents/
    └── ESRIS.pdf
```

*The structure may vary depending on the files included in the final repository.*

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/akhilm0001/IntelliResearch-AI.git
```

```bash
cd IntelliResearch-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv .researchai
```

Activate the environment on Windows:

```bash
.researchai\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Configure Ollama

Download Ollama from:

https://ollama.com/

Pull the Gemma 4 model:

```bash
ollama pull gemma4
```

Make sure Ollama is running locally.

### 5. Configure Tavily API Key

Create an account at:

https://tavily.com/

Set your API key as an environment variable:

```powershell
$env:TAVILY_API_KEY="your_tavily_api_key"
```

Do not upload API keys or `.env` files containing secrets to GitHub.

### 6. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the application.

## 🔍 How It Works

1. The user provides a research question.
2. Relevant information is retrieved from the uploaded PDF documents using the RAG pipeline.
3. The Web Research Agent collects additional information through Tavily.
4. CrewAI coordinates the agents to analyze the collected information.
5. The Report Writer Agent generates the final research report.

## 🔮 Future Enhancements

- Support for multiple PDF document uploads.
- Improved source attribution and citation generation.
- Research report export in PDF format.
- Enhanced document retrieval and response accuracy.
- Additional research agents for specialized tasks.

## 👨‍💻 Developer

**Akhil M.**

- GitHub: https://github.com/akhilm0001
- LinkedIn: https://www.linkedin.com/in/akhil2003/

## 📄 License

This project is intended for educational, research, and portfolio purposes.

---

**IntelliResearch AI – Making Research Smarter with Agentic AI.**
