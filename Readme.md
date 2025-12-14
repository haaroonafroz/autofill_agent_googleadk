# Autofill Agent

An intelligent form-filling agent that uses your CV to automatically fill job application forms. It leverages **Vision-based PDF processing** (Docling) for layout understanding, **OpenAI** for reasoning, and **Qdrant** for secure, multi-tenant vector storage.

The system uses an **Extension-Native Architecture** where the browser extension handles page interaction, and the Python backend handles intelligence.

## Architecture

1.  **Frontend (Browser Extension)**:
    *   Injects into the active tab (LinkedIn, Workday, etc.).
    *   Captures the HTML form structure.
    *   Executes fill/click actions commanded by the backend.
    *   Manages User ID persistence.

2.  **Backend (FastAPI + Python)**:
    *   **PDF Processing**: Uses [Docling](https://github.com/DS4SD/docling) to convert CVs into structured Markdown.
    *   **RAG Engine**: Stores chunks in **Qdrant** with strict user isolation (`user_id`).
    *   **Reasoning Agent**: Analyzes HTML fields + CV data to decide *what* to fill (e.g., "Field #fname needs 'John'").

## Prerequisites

*   Python 3.10+ (Required for Docling)
*   [Qdrant](https://qdrant.tech/) Instance (Cloud or Local)
*   OpenAI API Key
*   Google Chrome / Edge / Brave

## Installation

### 1. Backend Setup
1.  **Clone the repository**.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If on Windows, run as Administrator once to allow Docling model downloads.*
3.  **Configuration**:
    Copy `env.example` to `.env` and fill in your details:
    ```bash
    cp env.example .env
    ```
    Required: `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`.

### 2. Frontend Setup (Extension)
1.  Open Chrome and navigate to `chrome://extensions`.
2.  Enable **Developer Mode** (top right).
3.  Click **Load unpacked**.
4.  Select the `frontend/` folder from this repository.

## Usage

### 1. Start the Server
Run the backend API:
```bash
uvicorn autofill_agent.server:app --reload
```
The server runs at `http://localhost:8000`.

### 2. Use the Extension
1.  Navigate to a job application page.
2.  Click the **Autofill Agent** extension icon.
3.  **Upload CV**: Select your PDF resume and click Upload. This indexes your data.
4.  **Start Autofill**: Click "Start".
    *   The extension sends the page HTML to the backend.
    *   The backend thinks and returns a list of actions.
    *   The extension fills the form fields in real-time.

## Project Structure

*   `autofill_agent/`
    *   `server.py`: FastAPI entrypoint. Handles `/process_page` and `/upload_cv`.
    *   `agent.py`: Core logic. Analyzes HTML and queries RAG to generate actions.
    *   `retrieve_info_from_pdf.py`: Qdrant interactions (Search & Indexing).
    *   `load_and_process_pdf.py`: Docling PDF converter.
*   `frontend/`
    *   `manifest.json`: Extension configuration.
    *   `popup.js`: Extension UI logic.
    *   `content_script.js`: DOM manipulation script (injected into web pages).

## Troubleshooting

*   **"Cannot read page. Refresh?"**: Refresh the target webpage. The extension only works on pages loaded *after* it was installed.
*   **Qdrant Errors**: Ensure your `.env` credentials are correct and your Qdrant instance is reachable.
*   **Windows Errors**: If you see `WinError 1314`, run your terminal as Administrator for the first run.
