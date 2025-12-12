import os
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain / OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Custom Modules
from .load_and_process_pdf import load_and_split_pdf
from .retrieve_info_from_pdf import RAGManager
from .analyze_web_form import analyze_form_structure
from .interact_with_web_page import BrowserInteractor

# Playwright
from playwright.async_api import async_playwright, Playwright, Browser, Page

load_dotenv()

class AutofillAgent:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=self.openai_api_key
        )
        
        self.rag_manager: Optional[RAGManager] = None
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.interactor: Optional[BrowserInteractor] = None
        
        # State
        self.current_form_fields = []
        self.pdf_processed = False

    async def start_browser(self):
        """Initializes the browser session."""
        if self.page and not self.page.is_closed():
            return
        
        print("Starting Playwright browser...")
        self.playwright = await async_playwright().start()
        # Headless=False is useful for debugging/demo, but strictly speaking user asked for "browser extension" feel 
        # which usually implies visibility. Let's keep it visible for now or headless=True based on preference.
        # User didn't specify headless preference, but "agent scraps" implies backend.
        self.browser = await self.playwright.chromium.launch(headless=False) 
        self.page = await self.browser.new_page()
        self.interactor = BrowserInteractor(self.page)
        print("Browser started.")

    async def close_browser(self):
        """Clean up browser resources."""
        if self.page: await self.page.close()
        if self.browser: await self.browser.close()
        if self.playwright: await self.playwright.stop()
        print("Browser resources closed.")

    def process_pdf(self, pdf_path: str):
        """Loads and indexes the PDF."""
        print(f"Processing PDF: {pdf_path}")
        chunks = load_and_split_pdf(pdf_path)
        self.rag_manager = RAGManager() # Re-initializes connection
        self.rag_manager.initialize_vector_store(chunks, force_recreate=True)
        self.pdf_processed = True
        print("PDF processed and indexed in Qdrant.")

    async def fill_form(self, url: str):
        """Main logic to navigate, analyze, and fill the form."""
        if not self.interactor:
            await self.start_browser()
        
        if not self.pdf_processed:
            raise RuntimeError("PDF not processed. Call process_pdf() first.")

        # 1. Navigate
        print(f"Navigating to {url}...")
        await self.page.goto(url, wait_until='domcontentloaded')
        
        # 2. Analyze
        print("Analyzing form structure...")
        html_content = await self.interactor.get_page_content()
        fields = analyze_form_structure(html_content)
        self.current_form_fields = fields
        print(f"Found {len(fields)} fields.")

        # 3. Fill Loop
        for field in fields:
            await self._process_single_field(field)
            
        print("Form filling attempt complete.")

    async def _process_single_field(self, field: Dict[str, Any]):
        """Decides how to fill a single field using LLM + RAG."""
        label = field.get('label')
        name = field.get('name')
        f_type = field.get('type')
        selector = field.get('selector')
        options = field.get('options', [])

        if not selector or f_type in ['hidden', 'submit', 'button', 'image', 'reset']:
            return

        print(f"Processing field: Label='{label}', Name='{name}', Type='{f_type}'")

        # 1. Query RAG
        # Construct a query based on the field metadata
        query_prompt = f"What is the {label or name}?"
        if f_type == 'radio' or f_type == 'checkbox':
             query_prompt = f"Should I check the box for {label or name}?"

        context_docs = self.rag_manager.query_vector_store(query_prompt, k=3)
        context_text = "\n".join([doc.page_content for doc in context_docs])

        # 2. Ask LLM for the value
        system_prompt = """You are a helpful assistant filling out a job application form based on a user's CV.
        You will be given information from the CV and details about a form field.
        Your goal is to provide the exact value to fill into the field.
        
        - For text fields, return the text.
        - For radio/checkbox, return 'true' if it should be checked, 'false' otherwise.
        - For select/dropdown, return the EXACT option text from the provided list that matches the CV info.
        - If the information is not in the CV, return 'SKIP'.
        """
        
        user_message = f"""
        Field Label: {label}
        Field Name: {name}
        Field Type: {f_type}
        Options (if dropdown): {options}
        
        CV Context:
        {context_text}
        
        What value should I put in this field?
        """

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])
        
        value_to_fill = response.content.strip()
        
        if value_to_fill == 'SKIP':
            print(f"Skipping field {label or name} (Info not found).")
            return

        # 3. Execute Action
        try:
            if f_type == 'select' or (f_type == 'select-one' and options):
                # The LLM should have selected one of the options.
                # We try to match it.
                await self.interactor.select_dropdown_option(selector, label=value_to_fill)
            elif f_type in ['checkbox', 'radio']:
                should_check = value_to_fill.lower() == 'true'
                if should_check: # Only act if we need to check it (usually)
                    await self.interactor.set_checkbox(selector, should_check)
            else:
                # Text input
                await self.interactor.fill_field(selector, value_to_fill)
        except Exception as e:
            print(f"Failed to fill field {selector}: {e}")

# Example standalone run (if executed as script)
async def main():
    # Example usage
    agent = AutofillAgent()
    # Assume PDF and URL are provided via args or env for testing
    # agent.process_pdf("path/to/resume.pdf")
    # await agent.start_browser()
    # await agent.fill_form("https://jobs.example.com/apply")
    # await agent.close_browser()
    print("Agent initialized. Use via API.")

if __name__ == "__main__":
    asyncio.run(main())
