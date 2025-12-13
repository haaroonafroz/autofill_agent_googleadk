import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Custom Modules
from .load_and_process_pdf import load_and_split_pdf
from .retrieve_info_from_pdf import RAGManager
from .analyze_web_form import analyze_form_structure

load_dotenv()

class AutofillAgent:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        # 1. The Brain (LLM)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=self.openai_api_key
        )
        
        # 2. State & Resources
        self.rag_manager = RAGManager()

    def process_pdf(self, pdf_path: str, user_id: str):
        """Standard PDF loading (Non-Agentic setup step)."""
        print(f"Processing PDF for user {user_id}")
        chunks = load_and_split_pdf(pdf_path)
        self.rag_manager.initialize_vector_store(chunks, user_id=user_id, force_recreate=False)

    async def generate_form_actions(self, html_content: str, user_id: str) -> List[Dict]:
        """
        Analyzes HTML and returns a list of actions for the frontend to execute.
        This effectively replaces the "Autonomous Loop" with a single reasoning pass per page state.
        """
        # 1. Analyze HTML
        print(f"Analyzing HTML for user {user_id}...")
        fields = analyze_form_structure(html_content)
        print(f"Found {len(fields)} fields.")
        
        actions = []
        for field in fields:
            # 2. Query RAG & LLM for each field
            value = await self._get_value_for_field(field, user_id)
            
            if value and value != 'SKIP':
                # Map field type to action type
                action_type = "fill"
                if field['type'] in ['checkbox', 'radio']:
                    action_type = "check" if value.lower() == 'true' else "uncheck"
                    # For uncheck, we still send "check" action with value "false" typically, 
                    # but let's stick to our content_script logic: value='true'/'false'
                    value = value.lower()
                elif field['type'] == 'select' or field['type'] == 'select-one':
                    action_type = "select"

                actions.append({
                    "selector": field['selector'],
                    "action": action_type,
                    "value": value,
                    "type": field['type']
                })
        
        print(f"Generated {len(actions)} actions.")
        return actions

    async def _get_value_for_field(self, field: Dict[str, Any], user_id: str) -> str:
        """
        Internal method to determine the value for a single field.
        """
        label = field.get('label')
        name = field.get('name')
        f_type = field.get('type')
        options = field.get('options', [])

        if f_type in ['hidden', 'submit', 'button', 'image', 'reset']:
            return 'SKIP'

        # 1. Query RAG
        query_prompt = f"What is the {label or name}?"
        if f_type == 'radio' or f_type == 'checkbox':
             query_prompt = f"Should I check the box for {label or name}?"

        context_docs = self.rag_manager.query_vector_store(query_prompt, user_id=user_id, k=3)
        context_text = "\n".join([doc.page_content for doc in context_docs])

        # 2. Ask LLM
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

        response = await self.llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])
        
        return response.content.strip()
