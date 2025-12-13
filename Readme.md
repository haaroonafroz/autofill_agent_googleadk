# Autofill Agent

An intelligent form-filling agent that uses your CV to automatically fill job application forms. It leverages **Vision-based PDF processing** (Docling) for layout understanding, **OpenAI** for reasoning, and **Qdrant** for secure, multi-tenant vector storage.

## Architecture

*   **Backend Agent**: Python-based service using LangChain, OpenAI, and Playwright.
*   **PDF Processing**: Uses [Docling](https://github.com/DS4SD/docling) to convert PDFs into structured Markdown, preserving headers and multi-column layouts (crucial for Resumes).
*   **Vector Database**: Qdrant (Cloud or Local) stores chunks with strict tenant separation (`user_id`).
*   **Web Interaction**: Playwright for analyzing and filling form fields.
*   **API**: FastAPI server to expose the agent's capabilities.

## Prerequisites

*   Python 3.10+ (Required for Docling)
*   [Qdrant](https://qdrant.tech/) Instance (Cloud or Local)
*   OpenAI API Key

## Installation

1.  **Clone the repository**
2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This will install `docling`, which may download PyTorch libraries (~2GB).*

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
Send a POST request to upload and index your CV. This will:
1.  Convert the PDF to Markdown using Vision models.
2.  Split the text by Headers (Experience, Education, etc.).
3.  Index the chunks into Qdrant with your unique `user_id`.

```bash
# Example using curl
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
    *   `retrieve_info_from_pdf.py`: RAG pipeline (Qdrant + OpenAI Embeddings) with multi-tenant filtering.
    *   `load_and_process_pdf.py`: **New:** Docling-based PDF-to-Markdown converter.
    *   `analyze_web_form.py`: HTML form parsing.
    *   `interact_with_web_page.py`: Browser actions helper.

## Qdrant Configuration
The agent automatically creates a collection with a Payload Index for `user_id`, `Header 1`, and `Header 2`. This ensures:
1.  **Security**: Users can only query their own data (via `user_id` filter).
2.  **Precision**: The agent can search specifically within sections (e.g., "Find Python in 'Experience'").
