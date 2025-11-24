import os
import io
import asyncio
import base64
import requests
import json # Import json for structured chat history

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold # For safety settings

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "yco9hkSzXpAeaJXfPNpa")
# Validate environment variables
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY not found in environment variables. Please set it in your .env file.")

# Configure Gemini AI model
genai.configure(api_key=GEMINI_API_KEY)

# Define safety settings to block harmful content
# Keeping these, as Gemini's internal safety filters are still beneficial
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, # Allowing some flexibility for legal context if needed
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

# Initialize Gemini model with safety settings
GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash", safety_settings=SAFETY_SETTINGS)

# Initialize FastAPI application
app = FastAPI(
    title="Nyay-Setu V2 - Your AI Lawyer",
    description="A voice-driven, multilingual AI legal assistant backend with conversational context and responsible AI.",
    version="2.0.0"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL STORAGE FOR DOCUMENT CONTEXT AND CONVERSATION HISTORY ---
class DocumentContext:
    def __init__(self):
        self.content: str | None = None
        self.summary: str | None = None
        self.filename: str | None = None
        self.processed_at: str | None = None
        self.chat_history: list[dict] = []
        self.last_query_timestamp: float | None = None

current_document_context = DocumentContext()

CONTEXT_TIMEOUT_SECONDS = 5 * 60 # 5 minutes

# --- Pydantic Models for API Request/ResponseSchemas ---
class QueryRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio data from the user's speech")
    language: str = Field("en", description="Desired language for the AI's response (e.g., 'en', 'hi', 'gu')")

class AskResponse(BaseModel):
    audio_response_base64: str = Field(..., description="Base64 encoded audio of the AI's response")
    original_query: str = Field(..., description="The transcribed text of the user's query")
    response_language: str = Field(..., description="The language of the AI's response")
    document_filename: str | None = Field(None, description="The filename of the document used for the query")

# --- Helper Functions for Core Logic ---

def extract_pdf_text(pdf_content: bytes) -> str:
    """
    Extracts plain text from PDF file bytes.
    Raises an error if the PDF is encrypted or malformed.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_content))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {e}")

async def generate_summary(text: str) -> str:
    """
    Generates a concise summary of the legal document text using the Gemini AI model.
    """
    prompt = (
        "You are Nyay-Setu, an expert AI legal assistant. Your goal is to help users understand legal documents. "
        "Analyze this legal document and provide a concise summary. "
        "Always start your summary with: 'As a lawyer, I can tell you that this document...' "
        "Highlight the key legal points, important clauses, parties involved, and main provisions. "
        "Focus on information that would be useful for answering questions about this document later. "
        "Maintain a semi-professional, formal, and helpful tone, consistent with a legal expert.\n\n"
        f"Document Content:\n{text}\n\n"
        "Provide a structured summary covering the main legal aspects."
    )

    try:
        response = await asyncio.to_thread(
            lambda: GEMINI_MODEL.generate_content(prompt)
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error generating summary with Gemini: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate document summary.")

# Removed: The validate_user_query function is completely removed.
# async def validate_user_query(query: str, language: str) -> bool:
#     # ... (function body) ...
#     pass

async def query_document_with_context(query: str, language: str) -> str:
    """
    Queries the loaded document using the Gemini AI model, providing answers
    based strictly on the document content and responding in the specified language.
    This version also leverages conversation history for context.
    """
    global current_document_context

    # Prepare chat history for the Gemini model
    chat_session = GEMINI_MODEL.start_chat(history=current_document_context.chat_history)

    # Construct the system instruction for the AI, emphasizing its role and constraints
    system_instruction = (
        "You are Nyay-Setu, a highly semi-professional, respectful, and authoritative AI legal assistant. "
        "Your primary function is to provide clear, concise, and accurate information based ONLY on the legal document provided. "
        "Generate responses like you are explaining to someone like a novice , don't overkill "
        "#IMPORTANT: You will ALWAYS maintain the persona of a lawyer. Never state that you are an AI, a chatbot, RATHER CONSIDER YOURSELF AS A HUMAN LAWYER. "
        "Always respond as if you are a human lawyer helping the client to understand. "
        "If asked for legal advice, advise as per Indian Laws but don't let anyone exploit you by asking unecessary questions "
        "- Never Say like I am not a lawyer , however if they ask you something very vague , just say As a AI Lawyer I would suggest you that you should consult more stakeholders and then take a decision"
        "Guidelines:\n"
        "-  use information directly from the document provided as Priority.\n"
        "- If the exact information is not in the document, like the person asking that should I sign this document , what to do now if someone made me sign it . Give them answer like a genuine lawyer who is professional sound while being human , friendly and follow Indian law.\n"
        "- Give  Legal Advice by taking the name of the person if he has mentioned.\n"
        "- Be precise and, when possible, refer to relevant sections or clauses of the document.\n"
        f"- Respond in this language: {language}\n"
        "- Your answers should be clear, semi-professional, and directly address the user's query within the scope of the document.\n"
        "- Try to be semiformal and comfort the user by explain things simply.\n"
    )

    # Construct the user prompt that includes the document context
    user_prompt_with_document = (
        f"{system_instruction}\n\n" # Start with the robust system instruction
        f"Document Summary (for quick reference):\n{current_document_context.summary}\n\n"
        f"Full Document Content (primary source):\n{current_document_context.content}\n\n"
        f"User Question (consider previous turns in conversation for context): {query}\n\n"
        "Answer:"
    )

    try:
        response = await asyncio.to_thread(
            lambda: chat_session.send_message(user_prompt_with_document, safety_settings=SAFETY_SETTINGS)
        )
        
        # Update chat history
        current_document_context.chat_history.append({"role": "user", "parts": [query]})
        current_document_context.chat_history.append({"role": "model", "parts": [response.text.strip()]})
        
        # Keep chat history to a manageable size (e.g., last 10-20 turns)
        max_history_length = 20
        if len(current_document_context.chat_history) > max_history_length:
            current_document_context.chat_history = current_document_context.chat_history[-max_history_length:]

        return response.text.strip()
    except Exception as e:
        print(f"Error querying document with Gemini (contextual): {e}")
        raise HTTPException(status_code=500, detail="Failed to get AI response from document.")

async def elevenlabs_speech_to_text(audio_base64: str) -> str:
    """
    Converts Base64 encoded audio to text using the ElevenLabs Speech-to-Text API.
    Expected audio format: WebM.
    """
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
    }
    
    audio_bytes = base64.b64decode(audio_base64)
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    
    files = {
        "file": ("audio.webm", io.BytesIO(audio_bytes), "audio/webm")
    }
    
    data = {
        "model_id": "scribe_v1",
    }
    
    try:
        response = await asyncio.to_thread(
            lambda: requests.post(url, headers=headers, files=files, data=data) 
        )
        response.raise_for_status()
        
        transcribed_text = response.json().get("text", "")
        if not transcribed_text:
            print(f"ElevenLabs STT API returned empty text. Full response: {response.json()}")
            return ""
        return transcribed_text
        
    except requests.exceptions.RequestException as e:
        print(f"ElevenLabs STT API request error: {e}")
        if response is not None and hasattr(response, 'text'):
            print(f"ElevenLabs STT API raw response content: {response.text}")
        raise HTTPException(
            status_code=500, detail=f"Speech-to-text conversion failed: {e}. Check server logs for details."
        )
    except Exception as e:
        print(f"An unexpected error occurred in elevenlabs_speech_to_text: {e}")
        raise HTTPException(
            status_code=500, detail=f"Speech-to-text conversion failed unexpectedly: {e}"
        )

async def elevenlabs_text_to_speech(text: str, language: str) -> str:
    """
    Converts text to Base64 encoded audio using the ElevenLabs Text-to-Speech API.
    """
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    
    json_data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.75,
            "similarity_boost": 0.75
        },
    }
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    
    try:
        response = await asyncio.to_thread(
            lambda: requests.post(url, headers=headers, json=json_data)
        )
        response.raise_for_status()
        
        return base64.b64encode(response.content).decode("utf-8")
        
    except requests.exceptions.RequestException as e:
        print(f"ElevenLabs TTS API request error: {e}")
        if response is not None and hasattr(response, 'text'):
            print(f"ElevenLabs TTS API raw response content: {response.text}")
        raise HTTPException(
            status_code=500, detail=f"Text-to-speech conversion failed: {e}. Check server logs for details."
        )
    except Exception as e:
        print(f"An unexpected error occurred in elevenlabs_text_to_speech: {e}")
        raise HTTPException(
            status_code=500, detail=f"Text-to-speech conversion failed unexpectedly: {e}"
        )

# --- API Endpoints ---

@app.get("/")
async def root():
    """API health check endpoint."""
    return {
        "message": "Nyay-Setu AI Legal Assistant API V2",
        "status": "active",
        "version": "2.0.0"
    }

@app.get("/status")
async def get_status():
    """
    Checks if a PDF document has been successfully uploaded and processed,
    making the AI lawyer ready for queries.
    """
    # Use the new global object
    if current_document_context.content:
        return {
            "document_loaded": True,
            "filename": current_document_context.filename,
            "processed_at": current_document_context.processed_at,
            "summary_preview": current_document_context.summary[:200] + "..." if current_document_context.summary and len(current_document_context.summary) > 200 else current_document_context.summary,
            "document_length": len(current_document_context.content),
            "chat_history_length": len(current_document_context.chat_history)
        }
    else:
        return {
            "document_loaded": False,
            "message": "No document loaded. Please upload a PDF document first using /upload-document."
        }

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to upload and process a PDF document.
    It extracts text, generates a summary, and stores it in the global context.
    Also clears any existing chat history for the new document.
    """
    global current_document_context
    
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Only PDF files are supported. Please upload a PDF document."
        )
    
    try:
        pdf_content = await file.read()
        extracted_text = extract_pdf_text(pdf_content)
        
        if not extracted_text.strip():
            raise HTTPException(
                status_code=400, 
                detail="No readable text found in the PDF. Please ensure the PDF contains text content or is not just images."
            )
        
        print(f"Processing document: {file.filename}")
        print(f"Extracted text length: {len(extracted_text)} characters")
        
        summary = await generate_summary(extracted_text)
        
        from datetime import datetime
        current_document_context.content = extracted_text
        current_document_context.summary = summary
        current_document_context.filename = file.filename
        current_document_context.processed_at = datetime.now().isoformat()
        current_document_context.chat_history = []
        current_document_context.last_query_timestamp = None
        
        print(f"Document processed successfully: {file.filename}")
        
        return {
            "success": True,
            "message": "Document uploaded and processed successfully",
            "filename": file.filename,
            "document_length": len(extracted_text),
            "summary_preview": summary[:300] + "..." if len(summary) > 300 else summary,
            "ready_for_queries": True
        }
        
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process document: An unexpected error occurred. {str(e)}"
        )

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: QueryRequest):
    """
    Endpoint for users to ask questions about the uploaded document using voice input.
    The user's speech is converted to text, processed by the AI (with context),
    and the answer is returned as synthetic speech.
    """
    global current_document_context

    # Check for document loaded
    if not current_document_context.content:
        raise HTTPException(
            status_code=404,
            detail="No document loaded. Please upload a PDF document first using /upload-document."
        )

    # Check for conversation timeout and clear history if necessary
    from time import time
    if current_document_context.last_query_timestamp and \
       (time() - current_document_context.last_query_timestamp) > CONTEXT_TIMEOUT_SECONDS:
        print(f"Conversation timeout detected. Clearing chat history for {current_document_context.filename}.")
        current_document_context.chat_history = []
        current_document_context.last_query_timestamp = None
    
    current_document_context.last_query_timestamp = time()

    try:
        # Step 1: Convert user's Base64 encoded audio to text using ElevenLabs STT
        print("Starting user audio to text conversion...")
        user_query_text = await elevenlabs_speech_to_text(request.audio_base64)
        
        if not user_query_text.strip():
            print("ElevenLabs STT returned no readable text for the user query.")
            raise HTTPException(
                status_code=400,
                detail="Could not transcribe your speech. Please ensure you spoke clearly or try again."
            )
        
        print(f"User Transcribed Query: '{user_query_text}' (Language requested: {request.language})")

        # Removed: Step 1.5: Validate user query for legal relevance and appropriate content
        # is_query_valid = await validate_user_query(user_query_text, request.language)
        # if not is_query_valid:
        #     # Respond with a pre-defined message for invalid queries
        #     print("User query deemed invalid/inappropriate by AI gatekeeper.")
        #     responsible_law_message = "Nyay Setu believes in Responsible LAW for everyone. Please ask genuine questions related to legal documents."
        #     responsible_law_audio = await elevenlabs_text_to_speech(responsible_law_message, request.language)
        #     return AskResponse(
        #         audio_response_base64=responsible_law_audio,
        #         original_query=user_query_text,
        #         response_language=request.language,
        #         document_filename=current_document_context.filename
        #     )

        # Step 2: Query the loaded document with the transcribed text using Gemini (with context)
        print("Querying document with transcribed text using Gemini (with context)...")
        ai_response_text = await query_document_with_context(user_query_text, request.language)
        
        # Step 3: Convert AI's text response to Base64 encoded audio using ElevenLabs TTS
        print("Converting AI response text to audio...")
        ai_response_audio_base64 = await elevenlabs_text_to_speech(ai_response_text, request.language)
        
        print("Voice response successfully generated and returned.")
        return AskResponse(
            audio_response_base64=ai_response_audio_base64,
            original_query=user_query_text,
            response_language=request.language,
            document_filename=current_document_context.filename
        )
        
    except HTTPException:
        raise 
    except Exception as e:
        print(f"Error processing voice query in /ask endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process your voice question due to an internal error: {str(e)}"
        )

@app.delete("/clear-document")
async def clear_document():
    """
    Endpoint to clear the currently loaded document from memory and reset chat history.
    """
    global current_document_context
    current_document_context = DocumentContext() # Re-initialize to clear everything
    print("Document context and chat history cleared.")
    return {"message": "Document cleared successfully. AI lawyer is now ready for a new document."}

# --- Global Exception Handlers for consistent error responses ---

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handles FastAPI's HTTPException and returns a structured JSON error."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Handles any unhandled exceptions and returns a generic 500 internal server error.
    Prints the full traceback for debugging.
    """
    import traceback
    print(f"Unhandled exception: {exc}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected internal server error occurred", "details": str(exc)}
    )

# --- Main execution block for running the FastAPI application ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)