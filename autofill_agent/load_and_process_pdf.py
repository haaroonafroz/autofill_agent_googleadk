import os
from typing import List
from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Import Docling
try:
    from docling.document_converter import DocumentConverter
except ImportError:
    print("Warning: Docling not installed. Install it via 'pip install docling'")
    DocumentConverter = None

def load_and_split_pdf(pdf_file_path: str) -> List[Document]:
    """
    Loads a PDF using Docling (Vision/Layout aware) to convert it to Markdown,
    then splits it by semantic headers (e.g., # Experience, # Education).
    """
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_file_path}")

    if DocumentConverter is None:
        raise ImportError("Docling is required for this function. Please install it.")

    print(f"Converting PDF to Markdown using Docling: {pdf_file_path}")
    
    try:
        # 1. Convert PDF to Markdown using Docling
        # This handles multi-column layouts and table parsing automatically.
        converter = DocumentConverter()
        result = converter.convert(pdf_file_path)
        markdown_text = result.document.export_to_markdown()
        
        print("PDF converted to Markdown. Splitting by headers...")

        # 2. Split by Headers (Semantic Chunking)
        # This ensures "Experience" content stays with the "Experience" header.
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        header_splits = markdown_splitter.split_text(markdown_text)

        # 3. Recursive Split (Safety Net)
        # If a section (like a very long job description) is still too big, 
        # split it further while keeping the header metadata.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        
        final_chunks = text_splitter.split_documents(header_splits)
        
        # Add source metadata
        for chunk in final_chunks:
            chunk.metadata["source"] = pdf_file_path
            # The 'Header 1', 'Header 2' etc. metadata is already added by MarkdownHeaderTextSplitter
            
        print(f"Successfully processed PDF into {len(final_chunks)} structured chunks.")
        return final_chunks

    except Exception as e:
        print(f"Error processing PDF with Docling: {e}")
        raise

if __name__ == '__main__':
    # Test block
    test_pdf = "example_cv.pdf"
    if os.path.exists(test_pdf):
        try:
            chunks = load_and_split_pdf(test_pdf)
            for i, chunk in enumerate(chunks[:3]):
                print(f"\n--- Chunk {i} ---")
                print(f"Metadata: {chunk.metadata}")
                print(f"Content: {chunk.page_content[:200]}...")
        except Exception as e:
            print(f"Test failed: {e}")
