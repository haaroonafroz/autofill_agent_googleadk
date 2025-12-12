# Autofill Agent

An intelligent form-filling agent that uses your CV to automatically fill job application forms. It leverages OpenAI's LLMs for reasoning and Qdrant for semantic search over your CV data.

## Architecture

*   **Backend Agent**: Python-based service using LangChain, OpenAI, and Playwright.
*   **Vector Database**: Qdrant (Cloud or Local) to store and retrieve CV information.
*   **Web Interaction**: Playwright for analyzing and filling form fields.
*   **API**: FastAPI server to expose the agent's capabilities.

## Prerequisites

*   Python 3.10+
*   [Qdrant](https://qdrant.tech/) Instance (Cloud or Local)
*   OpenAI API Key

## Installation

1.  **Clone the repository**
2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Install Playwright Browsers**
    ```bash
    playwright install
    ```
4.  **Configuration**
    Copy `env.example` to `.env` and fill in your details:
    ```bash
    cp env.example .env
    ```
    
    Required variables:
    *   `OPENAI_API_KEY`: Your OpenAI API key.
    *   `QDRANT_URL`: URL of your Qdrant instance.
    *   `QDRANT_API_KEY`: API Key for Qdrant.

## Usage

### 1. Run the API Server
Start the backend server which handles the agent logic:

```bash
uvicorn autofill_agent.server:app --reload
```

The API will be available at `http://localhost:8000`.

### 2. Upload your CV
Send a POST request to upload and index your CV:

```bash
curl -X POST -F "file=@/path/to/your/cv.pdf" http://localhost:8000/upload_cv
```

### 3. Fill a Form
Trigger the agent to fill a form at a specific URL:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"url": "https://jobs.example.com/apply"}' http://localhost:8000/fill_form
```

The agent will launch a browser (visible by default) and attempt to fill the form fields based on your CV data.

## Project Structure

*   `autofill_agent/`
    *   `agent.py`: Main agent logic (OpenAI + Playwright).
    *   `server.py`: FastAPI application.
    *   `retrieve_info_from_pdf.py`: RAG pipeline (Qdrant + OpenAI Embeddings).
    *   `load_and_process_pdf.py`: PDF text extraction.
    *   `analyze_web_form.py`: HTML form parsing.
    *   `interact_with_web_page.py`: Browser actions helper.

## Browser Extension Integration (Planned)
To use this as a browser extension:
1.  The extension should capture the current tab's URL.
2.  Send the URL to the `POST /fill_form` endpoint.
3.  Alternatively, for a more integrated experience, the extension can inject a script that communicates with the `agent` API to retrieve values field-by-field.
