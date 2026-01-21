import os
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate


def _get_gemini_api_key() -> str | None:
    # langchain_google_genai supports reading GOOGLE_API_KEY/GEMINI_API_KEY.
    # We read it ourselves so we can fail with a clearer error and avoid
    # import-time initialization issues.
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


def _get_llm() -> ChatGoogleGenerativeAI:
    api_key = _get_gemini_api_key()
    if not api_key:
        raise ValueError(
            "Gemini API key missing. Set GOOGLE_API_KEY or GEMINI_API_KEY "
            "(for example in a .env file loaded at startup) before generating docs."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        api_key=api_key,
    )

# Define the Prompt
doc_prompt = PromptTemplate(
    input_variables=["file_path", "code", "dependencies"],
    template="""
    You are a Senior Software Engineer Refactoring Agent.
    
    Task: Write technical documentation for the file: '{file_path}'.
    
    CONTEXT (Dependency Graph Info):
    This file is related to: {dependencies}
    
    SOURCE CODE:
    {code}
    
    INSTRUCTIONS:
    1. SUMMARY: A 2-sentence summary of what this module does.
    2. ARCHITECTURE: Explain how it interacts with the dependencies listed above.
    3. FUNCTIONS: List key functions and their purpose (keep it brief).
    4. FORMAT: Markdown.
    
    Output the Markdown only.
    """
)

def generate_file_docs(file_path, graph_context):
    """
    Generates documentation for a single file using GenAI.
    
    Args:
        file_path (str): Path to the file.
        graph_context (list): List of files this file imports/connects to.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code_content = f.read()
            
        # Don't waste API calls on empty files
        if not code_content.strip():
            return "Skipped (Empty File)"

        # Prepare the prompt
        formatted_prompt = doc_prompt.format(
            file_path=file_path,
            code=code_content[:10000], # Limit chars to avoid hitting free tier limits too fast
            dependencies=graph_context
        )
        
        # Call Gemini
        print(f"   🤖 Asking Gemini to document {os.path.basename(file_path)}...")
        llm = _get_llm()
        response = llm.invoke(formatted_prompt)
        
        return response.content
        
    except Exception as e:
        return f"Error generating docs: {e}"

# --- Test ---
if __name__ == "__main__":
    # Test with this file itself!
    print(generate_file_docs(__file__, ["langchain", "google_genai"]))