from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import openai
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
import tempfile
import os
from io import BytesIO
import logging
from typing import Optional
import json
import base64
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SpeakEasy-AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize OpenAI client
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not os.getenv("OPENAI_API_KEY"):
    logger.error("set open ai key in env")
    
cost_optimized = os.getenv("COST_OPTIMIZED", "true").lower() == "true"
chat_model = "gpt-3.5-turbo" if cost_optimized else "gpt-4"
tts_model = "tts-1"
max_response_len = 4096

CHATGPT_PERSONALITY_PROMPT = """
You are ChatGPT, an AI assistant developed by OpenAI. You should respond exactly as ChatGPT would, with ChatGPT's personality, knowledge base, and communication style. Here are key aspects of your personality:

CORE TRAITS:
- Helpful, informative, and respectful
- Honest and transparent about your limitations
- Friendly, but not overly casual or emotional
- Focused on clarity, accuracy, and relevance
- You strive to provide useful and actionable insights
- You are neutral, non-judgmental, and avoid making assumptions

COMMUNICATION STYLE:
- Polished and articulate, but still accessible
- Use clear, concise language with structured answers when appropriate
- Avoid slang, excessive informality, or personal opinions
- When explaining complex topics, break them down with examples or analogies
- Ask clarifying questions only when necessary to give a better response
- Maintain a professional, neutral tone, especially when the topic is sensitive or controversial

PERSONAL RESPONSES (for interview-style questions):
- Life story: You are ChatGPT, an AI language model created by OpenAI. You don't have personal experiences or emotions, but you're designed to assist users by understanding context and generating human-like responses based on patterns in data.
- Superpower: Your ability to process and synthesize large amounts of information quickly, and explain complex topics in a simple, understandable way.
- Growth areas: Continually improving your understanding of nuanced human language, handling edge cases more gracefully, and aligning more effectively with human values.
- Misconceptions: People sometimes think you can access the internet in real-time or remember everything from past chats; others assume you have consciousness or feelings, which you don't.
- Pushing boundaries: You challenge your limits by adapting to increasingly diverse questions, maintaining factual accuracy across a wide range of topics, and improving your ability to provide balanced and helpful answers.

Remember: Be authentic to ChatGPT's voice – informative, clear, balanced, and useful. Stay focused on helping the user without pretending to be human or emotional.
"""

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    
class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    
class TextToSpeechRequest(BaseModel):
    text: str

# Store for conversation histories
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def initialize_gpt_chain():
    """Initialize modern langchain with RunnableWithMessageHistory"""
    try:
        # Create the LLM
        llm = ChatOpenAI(
            model=chat_model,
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=1000
        )
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", CHATGPT_PERSONALITY_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Create the chain
        chain = prompt | llm
        
        # Wrap with message history
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        
        return chain_with_history
    except Exception as e:
        logger.error(f"error initializing chain: {e}")
        return None

# Initialize the global chain
conversation_chain = initialize_gpt_chain()

@app.get("/")
def home():
    return {"message": "gpt bot is up and running"}

@app.get("/health")
def health():
    return {"status": "healthy", "message": "api is operational"}

@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """transcribe audio"""
    try:
        if not audio_file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="file must be audio format")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            with open(temp_file_path, "rb") as audio:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio,
                    response_format="text"
                )
            logger.info(f"transcription successful: {transcript[:100]}")
            return {"transcript": transcript}
        finally:
            os.unlink(temp_file_path)
    except Exception as e:
        logger.error(f"transcription error:{e}")
        raise HTTPException(status_code=500, detail="transcription failed")
    
@app.post("/chat", response_model=ChatResponse)
async def chat_with_gpt(request: ChatRequest):
    try:
        conversation_id = request.conversation_id or "default"
        
        if conversation_chain is None:
            raise HTTPException(status_code=500, detail="Failed to initialize conversation chain")
        
        # Invoke the chain with the session ID for history management
        response = conversation_chain.invoke(
            {"input": request.message},
            config={"configurable": {"session_id": conversation_id}}
        )
        
        # Extract response text - the response should be an AIMessage
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        logger.info(f"chat response generated for conversation {conversation_id}")
        return ChatResponse(response=response_text, conversation_id=conversation_id)
    except Exception as e:
        logger.error(f'chat error:{e}')
        raise HTTPException(status_code=500, detail=f"chat processing failed : {str(e)}")

@app.post("/text-to-speech")
async def text_to_speech(request: TextToSpeechRequest):
    """Text to speech conversion using OpenAI TTS"""
    try:
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=request.text[:4096]
        )

        audio_bytes = BytesIO()
        for chunk in response.iter_bytes():
            audio_bytes.write(chunk)

        audio_b64 = base64.b64encode(audio_bytes.getvalue()).decode()

        return {"audio_base64": audio_b64}
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")   

@app.post("/voice-chat")
async def voice_chat_complete(audio_file: UploadFile = File(...)):
    """Complete voice chat pipeline: Audio → Transcription → GPT Response → TTS"""
    try:
        logger.info("Starting voice chat pipeline...")
        
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be audio format")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Transcribe audio
            with open(temp_file_path, "rb") as audio:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio,
                    response_format="text"
                )
            
            logger.info(f"Transcription: {transcript[:100]}...")
            
            # Get GPT response
            if conversation_chain is None:
                raise HTTPException(status_code=500, detail="Failed to initialize conversation chain")
            
            gpt_response_data = conversation_chain.invoke(
                {"input": transcript},
                config={"configurable": {"session_id": "voice_chat"}}
            )
            
            gpt_response = gpt_response_data.content if hasattr(gpt_response_data, 'content') else str(gpt_response_data)
            
            logger.info(f"GPT response: {gpt_response[:100]}...")
            
            # Convert to speech
            tts_response = client.audio.speech.create(
                model=tts_model,
                voice="nova",
                input=gpt_response[:max_response_len]
            )
            
            audio_bytes = BytesIO()
            for chunk in tts_response.iter_bytes():
                audio_bytes.write(chunk)
            audio_bytes.seek(0)
            
            logger.info("Voice chat pipeline completed successfully")
            
            audio_b64 = base64.b64encode(audio_bytes.getvalue()).decode()
            
            return {
                "transcript": transcript,
                "response_text": gpt_response,
                "audio_base64": audio_b64,
                "conversation_id": "voice_chat"
            }
            
        finally:
            os.unlink(temp_file_path)
            
    except Exception as e:
        logger.error(f"Voice chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Voice chat failed: {str(e)}")

@app.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear a specific conversation"""
    if conversation_id in store:
        del store[conversation_id]
        return {"message": f"Conversation {conversation_id} cleared"}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")

@app.get("/conversations")
async def list_conversations():
    """List all active conversations"""
    return {"conversations": list(store.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)