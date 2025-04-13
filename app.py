from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import pinecone utilities
try:
    from pinecone_utils import store_chat, search_similar
    pinecone_available = True
    logger.info("Pinecone integration available")
except ImportError as e:
    logger.warning(f"Pinecone integration not available - {e}")
    pinecone_available = False
    
    # Dummy functions if Pinecone is not available
    def store_chat(question, response):
        return False
    
    def search_similar(query, top_k=3):
        return []

# Initialize FastAPI app
app = FastAPI(title="Startup Buddy API", description="Backend API for Startup Buddy Q&A Chatbot")

# Add CORS middleware to allow requests from your Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Google Gemini
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash-002")
    gemini_available = True
    logger.info("Google Gemini API configured successfully")
except Exception as e:
    logger.error(f"Gemini API configuration failed - {e}")
    gemini_available = False

# Create a chat session storage
chat_sessions = {}

# Define request/response models
class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class SimilarResult(BaseModel):
    question: str
    response: str
    score: float

class ChatResponse(BaseModel):
    response: str
    session_id: str
    similar_results: List[SimilarResult] = []

class ConfigStatus(BaseModel):
    gemini_available: bool
    pinecone_available: bool
    google_api_key_configured: bool
    pinecone_api_key_configured: bool
    pinecone_index_name_configured: bool

class ChatHistoryItem(BaseModel):
    role: str
    content: str

class ChatHistoryResponse(BaseModel):
    history: List[ChatHistoryItem]

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint providing API information"""
    return {
        "message": "Welcome to Startup Buddy API",
        "version": "1.0",
        "status": "running",
        "endpoints": {
            "status": "/api/status",
            "chat": "/api/chat",
            "history": "/api/history/{session_id}"
        },
        "docs": "/docs"
    }

# Endpoints
@app.get("/api/status", response_model=ConfigStatus)
async def get_status():
    """Get the configuration status of the API"""
    return {
        "gemini_available": gemini_available,
        "pinecone_available": pinecone_available,
        "google_api_key_configured": bool(os.getenv("GOOGLE_API_KEY")),
        "pinecone_api_key_configured": bool(os.getenv("PINECONE_API_KEY")),
        "pinecone_index_name_configured": bool(os.getenv("PINECONE_INDEX_NAME"))
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: QuestionRequest):
    """Send a question to the chatbot and get a response"""
    if not gemini_available:
        raise HTTPException(status_code=503, detail="Gemini API is not available")
    
    # Create or retrieve chat session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in chat_sessions:
        chat_sessions[session_id] = model.start_chat(history=[])
    
    chat_session = chat_sessions[session_id]
    
    # Log incoming request
    logger.info(f"Received chat request with session_id: {session_id}")
    logger.info(f"Question: {request.question}")
    
    # Search for similar questions if Pinecone is available
    similar_results = []
    if pinecone_available:
        try:
            similar_docs = search_similar(request.question)
            similar_results = [
                {"question": doc["question"], "response": doc["response"], "score": doc["score"]}
                for doc in similar_docs
            ]
            logger.info(f"Found {len(similar_results)} similar questions")
        except Exception as e:
            logger.error(f"Error searching similar questions: {str(e)}")
    
    # Get response from Gemini
    try:
        response = chat_session.send_message(request.question)
        response_text = response.text
        
        # Store chat in Pinecone if available
        if pinecone_available:
            try:
                store_chat(request.question, response_text)
                logger.info("Chat stored in Pinecone successfully")
            except Exception as e:
                logger.error(f"Error storing chat in Pinecone: {str(e)}")
        
        return {
            "response": response_text,
            "session_id": session_id,
            "similar_results": similar_results
        }
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/api/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    """Get the chat history for a specific session"""
    if session_id not in chat_sessions:
        logger.warning(f"Chat session not found: {session_id}")
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    # Extract history from the chat session
    chat_session = chat_sessions[session_id]
    
    # Format history as a list of role/content pairs
    history = []
    
    # Extract history based on the structure of Google Gemini's chat history
    try:
        # Try accessing history property if it exists
        if hasattr(chat_session, 'history'):
            for message in chat_session.history:
                if hasattr(message, 'role') and hasattr(message, 'parts'):
                    # Extract content from parts
                    content = ' '.join([str(part) for part in message.parts])
                    history.append({"role": message.role, "content": content})
                elif isinstance(message, dict) and 'role' in message and 'parts' in message:
                    # Alternative structure
                    content = ' '.join([str(part) for part in message['parts']])
                    history.append({"role": message['role'], "content": content})
        logger.info(f"Retrieved history for session {session_id}, {len(history)} messages")
    except Exception as e:
        logger.error(f"Error extracting chat history: {str(e)}")
        # Fallback to empty history
        history = []
    
    return {"history": history}

# For testing purposes
@app.get("/api/test")
async def test_endpoint():
    """Simple test endpoint to verify API is working"""
    return {"status": "success", "message": "API is working properly"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import os
# import google.generativeai as genai
# from dotenv import load_dotenv
# from typing import List, Dict, Any, Optional
# import uuid
# import logging

# # LangChain imports for memory
# from langchain.memory import ConversationBufferWindowMemory, ChatMessageHistory
# from langchain.schema import HumanMessage, AIMessage, SystemMessage

# # Configure logging
# logging.basicConfig(level=logging.INFO, 
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# # Import pinecone utilities
# try:
#     from pinecone_utils import store_chat, search_similar
#     pinecone_available = True
#     logger.info("Pinecone integration available")
# except ImportError as e:
#     logger.warning(f"Pinecone integration not available - {e}")
#     pinecone_available = False
    
#     # Dummy functions if Pinecone is not available
#     def store_chat(question, response):
#         return False
    
#     def search_similar(query, top_k=3):
#         return []

# # Initialize FastAPI app
# app = FastAPI(title="Startup Buddy API", description="Backend API for Startup Buddy Q&A Chatbot")

# # Add CORS middleware to allow requests from your Next.js frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Update with your frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Configure Google Gemini
# try:
#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         raise ValueError("GOOGLE_API_KEY environment variable not set")
    
#     genai.configure(api_key=api_key)
#     model = genai.GenerativeModel("gemini-1.5-flash-002")
#     gemini_available = True
#     logger.info("Google Gemini API configured successfully")
# except Exception as e:
#     logger.error(f"Gemini API configuration failed - {e}")
#     gemini_available = False

# # Create a chat session storage
# chat_sessions = {}
# # Create a memory buffer storage
# memory_buffers = {}

# # Define request/response models
# class QuestionRequest(BaseModel):
#     question: str
#     session_id: Optional[str] = None

# class SimilarResult(BaseModel):
#     question: str
#     response: str
#     score: float

# class ChatResponse(BaseModel):
#     response: str
#     session_id: str
#     similar_results: List[SimilarResult] = []

# class ConfigStatus(BaseModel):
#     gemini_available: bool
#     pinecone_available: bool
#     google_api_key_configured: bool
#     pinecone_api_key_configured: bool
#     pinecone_index_name_configured: bool

# class ChatHistoryItem(BaseModel):
#     role: str
#     content: str

# class ChatHistoryResponse(BaseModel):
#     history: List[ChatHistoryItem]

# # Root endpoint
# @app.get("/")
# async def root():
#     """Root endpoint providing API information"""
#     return {
#         "message": "Welcome to Startup Buddy API",
#         "version": "1.0",
#         "status": "running",
#         "endpoints": {
#             "status": "/api/status",
#             "chat": "/api/chat",
#             "history": "/api/history/{session_id}"
#         },
#         "docs": "/docs"
#     }

# # Endpoints
# @app.get("/api/status", response_model=ConfigStatus)
# async def get_status():
#     """Get the configuration status of the API"""
#     return {
#         "gemini_available": gemini_available,
#         "pinecone_available": pinecone_available,
#         "google_api_key_configured": bool(os.getenv("GOOGLE_API_KEY")),
#         "pinecone_api_key_configured": bool(os.getenv("PINECONE_API_KEY")),
#         "pinecone_index_name_configured": bool(os.getenv("PINECONE_INDEX_NAME"))
#     }

# @app.post("/api/chat", response_model=ChatResponse)
# async def chat(request: QuestionRequest):
#     """Send a question to the chatbot and get a response"""
#     if not gemini_available:
#         raise HTTPException(status_code=503, detail="Gemini API is not available")
    
#     # Create or retrieve chat session
#     session_id = request.session_id or str(uuid.uuid4())
#     if session_id not in chat_sessions:
#         chat_sessions[session_id] = model.start_chat(history=[])
#         # Initialize the memory buffer for this session
#         chat_history = ChatMessageHistory()
#         memory_buffers[session_id] = ConversationBufferWindowMemory(
#             memory_key="history", 
#             chat_memory=chat_history, 
#             k=3,
#             return_messages=True
#         )
    
#     chat_session = chat_sessions[session_id]
#     memory = memory_buffers[session_id]
    
#     # Log incoming request
#     logger.info(f"Received chat request with session_id: {session_id}")
#     logger.info(f"Question: {request.question}")
    
#     # Search for similar questions if Pinecone is available
#     similar_results = []
#     if pinecone_available:
#         try:
#             similar_docs = search_similar(request.question)
#             similar_results = [
#                 {"question": doc["question"], "response": doc["response"], "score": doc["score"]}
#                 for doc in similar_docs
#             ]
#             logger.info(f"Found {len(similar_results)} similar questions")
#         except Exception as e:
#             logger.error(f"Error searching similar questions: {str(e)}")
    
#     # Get response from Gemini
#     try:
#         response = chat_session.send_message(request.question)
#         response_text = response.text
        
#         # Update the memory buffer directly
#         memory.chat_memory.add_user_message(request.question)
#         memory.chat_memory.add_ai_message(response_text)
        
#         logger.info(f"Added to memory buffer: Q: {request.question} A: {response_text[:30]}...")
        
#         # Store chat in Pinecone if available
#         if pinecone_available:
#             try:
#                 store_chat(request.question, response_text)
#                 logger.info("Chat stored in Pinecone successfully")
#             except Exception as e:
#                 logger.error(f"Error storing chat in Pinecone: {str(e)}")
        
#         return {
#             "response": response_text,
#             "session_id": session_id,
#             "similar_results": similar_results
#         }
#     except Exception as e:
#         logger.error(f"Error processing chat request: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# @app.get("/api/history/{session_id}", response_model=ChatHistoryResponse)
# async def get_chat_history(session_id: str):
#     """Get the chat history for a specific session"""
#     if session_id not in chat_sessions:
#         logger.warning(f"Chat session not found: {session_id}")
#         raise HTTPException(status_code=404, detail="Chat session not found")
    
#     history = []
    
#     # Get history from LangChain memory buffer if available
#     if session_id in memory_buffers:
#         try:
#             # Extract messages from LangChain memory
#             memory = memory_buffers[session_id]
            
#             # Get messages directly from chat_memory
#             messages = memory.chat_memory.messages
            
#             for message in messages:
#                 if isinstance(message, HumanMessage):
#                     history.append({"role": "user", "content": message.content})
#                 elif isinstance(message, AIMessage):
#                     history.append({"role": "assistant", "content": message.content})
#                 elif isinstance(message, SystemMessage):
#                     history.append({"role": "system", "content": message.content})
            
#             logger.info(f"Retrieved history from LangChain memory for session {session_id}, {len(history)} messages")
#         except Exception as e:
#             logger.error(f"Error extracting chat history from LangChain memory: {str(e)}")
    
#     # If LangChain memory failed or is empty, fallback to Gemini chat history
#     if not history:
#         try:
#             # Extract history based on the structure of Google Gemini's chat history
#             chat_session = chat_sessions[session_id]
#             if hasattr(chat_session, 'history'):
#                 for message in chat_session.history:
#                     if hasattr(message, 'role') and hasattr(message, 'parts'):
#                         # Extract content from parts
#                         content = ' '.join([str(part) for part in message.parts])
#                         history.append({"role": message.role, "content": content})
#                     elif isinstance(message, dict) and 'role' in message and 'parts' in message:
#                         # Alternative structure
#                         content = ' '.join([str(part) for part in message['parts']])
#                         history.append({"role": message['role'], "content": content})
#             logger.info(f"Retrieved history from Gemini for session {session_id}, {len(history)} messages")
#         except Exception as e:
#             logger.error(f"Error extracting chat history from Gemini: {str(e)}")
    
#     return {"history": history}

# # For testing purposes
# @app.get("/api/test")
# async def test_endpoint():
#     """Simple test endpoint to verify API is working"""
#     return {"status": "success", "message": "API is working properly"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
