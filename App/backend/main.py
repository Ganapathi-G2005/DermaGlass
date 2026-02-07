import os
import re
import logging
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from PIL import Image
import io
import json
import torch
import torch.nn as nn
from torchvision import models, transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# LangGraph & LangChain Imports
from typing import Annotated, List, Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="DermaDetect India API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # LangChain LLM wrapper
    llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it", google_api_key=GEMINI_API_KEY, temperature=0.3)
else:
    llm = None

# --- 1. Markdown Formatting Agent ---
def format_advice_markdown(raw: str) -> str:
    """Post-processes Gemini output into clean, well-structured Markdown."""
    if not raw or not isinstance(raw, str):
        return raw

    lines = raw.strip().split("\n")
    out = []
    in_blockquote = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        if re.match(r"^\d+\.\s+\*\*[^*]+\*\*", stripped):
            title = re.sub(r"^\d+\.\s+", "", stripped)
            out.append("")
            out.append(f"## {title}")
            continue
        if re.match(r"^#+\s", stripped):
            out.append(line)
            continue

        if re.match(r"^\*\*[^*]+\*\*:?\s*$", stripped) and not stripped.startswith("**CRITICAL"):
            title = stripped.strip("*").strip(":")
            out.append("")
            out.append(f"## {title}")
            continue

        if stripped.startswith(">"):
            if not in_blockquote:
                in_blockquote = True
            out.append(line)
            continue
        else:
            in_blockquote = False

        out.append(line)

    return "\n".join(out).strip()

# --- 2. Model Loading & Inference ---
MODEL_PATH = "models/best_model.pth"
CLASS_NAMES_PATH = "models/class_names.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
class_names = []

def load_ai_model():
    global model, class_names
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        logger.info(f"Loaded {len(class_names)} classes.")
    else:
        logger.error("ERROR: class_names.json not found!")
        return

    logger.info("Loading EfficientNet-B0...")
    model = models.efficientnet_b0(weights=None)
    num_classes = len(class_names)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        logger.info("Model loaded successfully!")
    else:
        logger.error(f"ERROR: Model file {MODEL_PATH} not found!")

load_ai_model()

inference_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def run_custom_model_inference(image_bytes):
    if model is None:
        return "Model Error", 0.0
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = inference_transforms(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        confidence_score = confidence.item() * 100.0
        predicted_class_idx = predicted_idx.item()
        
        if predicted_class_idx < len(class_names):
            predicted_label = class_names[predicted_class_idx]
            # Clean up label (remove leading numbers like "11 Melasma" -> "Melasma")
            predicted_label = re.sub(r'^\d+\s+', '', predicted_label)
        else:
            predicted_label = "Unknown"
        
        # --- NEW THRESHOLD LOGIC (User Request) ---
        if confidence_score < 50.0:
            logger.info(f"Inference: Low Confidence ({confidence_score:.2f}%) -> Returning Unclear/Normal")
            # We return a special label that the analysis node will handle
            return "Unclear / Normal", confidence_score
            
        logger.info(f"Inference: {predicted_label} ({confidence_score:.2f}%)")
        return predicted_label, confidence_score
    except Exception as e:
        logger.error(f"Inference Error: {e}")
        return "Error", 0.0

# --- 3. LangGraph Agentic Workflow ---

# Define State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The chat history"]
    disease: str
    confidence: float
    analysis_report: str # The initial report generated

# Node 1: Analysis Reporter
def analysis_reporter_node(state: AgentState):
    """
    Generates the initial medical analysis report.
    This acts as the 'Start' node logic when a new prediction is made.
    """
    disease = state["disease"]
    confidence = state["confidence"]
    
    # Logic from original code for low confidence
    # Logic from original code for low confidence
    if disease == "Unclear / Normal" or confidence < 50.0:
        advice = (
            "## Analysis Inconclusive\n"
            "The image is unclear or does not strongly match any specific skin condition in my training.\n\n"
            "**Possibilities:**\n"
            "- It might be **normal healthy skin**.\n"
            "- The image might be **blurry or poorly lit**.\n"
            "- The condition might be outside my current knowledge base.\n\n"
            "**Recommendation:**\n"
            "Please try capturing the image again in better lighting. If you have concerns, always consult a dermatologist."
        )
        return {"analysis_report": advice}
    
    if confidence <= 60.0:
        prefix = "**Note: Confidence is moderate.**\n\n"
    else:
        prefix = ""

    prompt = (
        f"My diagnostic model detected **{disease}** with **{confidence:.1f}%** confidence. "
        "Act as a helpful AI dermatology assistant for an Indian patient. "
        "Provide your response in STRICT Markdown format:\n\n"
        "**FORMATTING RULES:**\n"
        "- Use `## ` for each main section heading (e.g. `## What is it?`, `## Immediate Care`).\n"
        "- Use **bold** for important terms, medication names, and key warnings.\n"
        "- Use > blockquotes for critical warnings (e.g. steroid creams, when to see a doctor).\n"
        "- Use bullet points (- or *) for lists.\n"
        "- Keep sections short: 2–4 sentences each.\n\n"
        "**SECTIONS (use ## for each):**\n"
        "1. **What is it?** – Simple, non-medical explanation.\n"
        "2. **Immediate Care** – Safe first-line care. If suggesting medication, use **bold** for generics only (e.g. **Benzoyl Peroxide**, **Clotrimazole**).\n"
        "3. **Home Remedies** – 2–3 safe Indian household tips if applicable.\n"
        "4. **Precaution** – One hygiene tip to prevent spreading.\n\n"
        "**CRITICAL:** Do NOT include a general medical disclaimer at the start, as the app displays one separately. "
        "However, DO put specific warning about steroid creams or contagiousness in a blockquote (>)."
    )

    if not llm:
         return {"analysis_report": "AI Service Unavailable"}

    try:
        response = llm.invoke(prompt)
        advice = format_advice_markdown(response.content)
        final_advice = prefix + advice
        # We also seed the history with a system message containing this context
        # Changed to HumanMessage because 'gemma-3-27b-it' threw "Developer instruction is not enabled"
        system_msg = HumanMessage(content=f"SYSTEM CONTEXT: You are a dermatology assistant. The user has been diagnosed with {disease} (Confidence: {confidence:.1f}%).\n\nOriginal Analysis:\n{final_advice}")
        return {"analysis_report": final_advice, "messages": [system_msg]}
    except Exception as e:
        return {"analysis_report": f"Error generating advice: {str(e)}"}

# Node 2: Assistant (Chat Loop)
def assistant_node(state: AgentState):
    """
    Handles follow-up questions using memory (messages in state).
    """
    if not llm:
        return {"messages": [AIMessage(content="AI Service Unavailable")]}
    
    # The state['messages'] automatically contains the history due to LangGraph's reducer (if using add_messages, but here we use simple list for TypedDict default behavior which is overwrite unless Annotated with add_messages. 
    # For this simple implementation, we assume 'messages' is the full history passed in update).
    # actually LangGraph StateGraph with Annotated[List, add_messages] is better, but TypedDict defaults to simple replacement if not using the reducer.
    # To keep it simple and robust for this demo:
    
    # We will just pass the messages to the LLM. 
    # Context is already in the SystemMessage from node 1.
    
    # ENFORCE PERSONA: Prepend a strict instruction to the messages sent to LLM.
    # We do NOT add this to state['messages'] to avoid bloating history, we just use it for the generation.
    
    guardrail_prompt = (
        "INSTRUCTIONS:\n"
        "You are Dr. Derma, a specialized Idian AI Dermatologist Assistant.\n"
        "Your GOAL is to help Indian users with their skin conditions, medication, and health advice.\n\n"
        "RULES:\n"
        "1. STAY ON TOPIC: You strictly ONLY answer questions related to dermatology, skin care, the specific diagnosis context, or general health.\n"
        "2. REFUSE IRRELEVANT QUESTIONS: If the user asks about math, coding, politics, movies, sports, or general knowledge, politely refuse. "
        "Say: \"I am a dermatology AI assistant. I can only help you with skin-related queries. Do you have any questions about your condition?\"\n"
        "3. TONE: Be professional, empathetic, and concise."
    )
    
    # We use HumanMessage because SystemMessage is rejected by the current Gemma endpoint
    messages_for_llm = [HumanMessage(content=guardrail_prompt)] + state["messages"]
    
    logger.info(f"--- Assistant Node: Invoking LLM with {len(messages_for_llm)} messages ---")
    try:
        response = llm.invoke(messages_for_llm)
        logger.info("--- Assistant Node: LLM Response Received ---")
        return {"messages": [response]} 
    except Exception as e:
        logger.error(f"!!! Assistant Node Error: {str(e)} !!!")
        import traceback
        traceback.print_exc()
        return {"messages": [AIMessage(content=f"I'm sorry, I encountered an error while processing your request: {str(e)}")]}

# Graph Setup
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    disease: str
    confidence: float
    analysis_report: str

# Define the graph logic
def router_node(state: AgentState):
    return state

def route_step(state: GraphState):
    if state.get("disease") and not state.get("analysis_report"): 
        return "analysis_reporter"
    return "assistant"

workflow = StateGraph(GraphState)
workflow.add_node("analysis_reporter", analysis_reporter_node)
workflow.add_node("assistant", assistant_node)

workflow.set_conditional_entry_point(
    route_step,
    {
        "analysis_reporter": "analysis_reporter",
        "assistant": "assistant"
    }
)
workflow.add_edge("analysis_reporter", END)
workflow.add_edge("assistant", END)

# Memory
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# --- 4. API Endpoints ---

class PredictionResponse(BaseModel):
    disease: str
    confidence: float
    advice: str
    thread_id: str 

class ChatRequest(BaseModel):
    disease: str
    confidence: float
    question: str

class ChatResponse(BaseModel):
    reply: str

@app.get("/")
def home():
    return {"message": "DermaDetect Agentic API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    print(f"--- Predict Request Received: {file.filename} ---")
    
    # 1. Inference
    contents = await file.read()
    print("Running Inference...")
    disease_name, confidence = run_custom_model_inference(contents)
    print(f"Inference Result: {disease_name} ({confidence:.2f}%)")
    
    # 2. Agentic Workflow - Phase 1: Analysis
    thread_id = "global_user_session"
    config = {"configurable": {"thread_id": thread_id}}
    
    inputs = {
        "disease": disease_name,
        "confidence": confidence,
        "analysis_report": "" # Force router to pick analysis
    }
    
    print("Invoking Analysis Graph...")
    try:
        events = graph.invoke(inputs, config=config)
        print("Graph Invocation Complete.")
        
        advice = events.get("analysis_report", "No advice generated.")
        print(f"Advice Generated (Length: {len(advice)})")
        
        return {
            "disease": disease_name,
            "confidence": round(confidence, 2),
            "advice": advice,
            "thread_id": thread_id
        }
    except Exception as e:
        print(f"Graph Error in Predict: {e}")
        import traceback
        traceback.print_exc()
        return {
             "disease": disease_name,
             "confidence": round(confidence, 2),
             "advice": f"Error generating advice: {str(e)}",
             "thread_id": thread_id
        }

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    print(f"Chat Request: {request.question}")
    try:
        thread_id = "global_user_session"
        config = {"configurable": {"thread_id": thread_id}}
        
        # 1. Update state with user message
        user_msg = HumanMessage(content=request.question)
        
        # 2. Invoke graph. 
        # Check current state for debugging
        cutoff_state = graph.get_state(config)
        print(f"Current State Check: {cutoff_state.values.keys() if cutoff_state else 'None'}")
        
        # If state is empty (server restart), we might want to ensure 'assistant' path is safe
        # But our router handles missing 'disease' by going to assistant.
        
        events = graph.invoke({"messages": [user_msg]}, config=config)
        print(f"Graph Events Keys: {events.keys()}")
        
        if "messages" in events and len(events["messages"]) > 0:
            response_msg = events["messages"][-1]
            print(f"AI Reply: {response_msg.content[:50]}...")
            return ChatResponse(reply=response_msg.content)
        else:
            print("Graph did not return messages.")
            return ChatResponse(reply="Error: No response from AI agent.")
            
    except Exception as e:
        print(f"Chat Endpoint Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return ChatResponse(reply=f"System Error: {str(e)}")

# --- 4. API Endpoints ---

class PredictionResponse(BaseModel):
    disease: str
    confidence: float
    advice: str
    thread_id: str # Added for frontend to track session if it wants (optional use)

class ChatRequest(BaseModel):
    disease: str
    confidence: float
    question: str

class ChatResponse(BaseModel):
    reply: str

@app.get("/")
def home():
    return {"message": "DermaDetect Agentic API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    logger.info(f"--- Predict Request Received: {file.filename} ---")
    
    # 1. Inference
    contents = await file.read()
    logger.info("Running Inference...")
    disease_name, confidence = run_custom_model_inference(contents)
    logger.info(f"Inference Result: {disease_name} ({confidence:.2f}%)")
    
    # 2. Agentic Workflow - Phase 1: Analysis
    thread_id = "global_user_session"
    config = {"configurable": {"thread_id": thread_id}}
    
    inputs = {
        "disease": disease_name,
        "confidence": confidence,
        "analysis_report": "" # Force router to pick analysis
    }
    
    logger.info("Invoking Analysis Graph...")
    try:
        events = graph.invoke(inputs, config=config)
        logger.info("Graph Invocation Complete.")
        
        advice = events.get("analysis_report", "No advice generated.")
        logger.info(f"Advice Generated (Length: {len(advice)})")
        
        return {
            "disease": disease_name,
            "confidence": round(confidence, 2),
            "advice": advice,
            "thread_id": thread_id
        }
    except Exception as e:
        logger.error(f"Graph Error in Predict: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
             "disease": disease_name,
             "confidence": round(confidence, 2),
             "advice": f"Error generating advice: {str(e)}",
             "thread_id": thread_id
        }

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    logger.info(f"Chat Request: {request.question}")
    try:
        thread_id = "global_user_session"
        config = {"configurable": {"thread_id": thread_id}}
        
        # 1. Update state with user message
        user_msg = HumanMessage(content=request.question)
        
        # 2. Invoke graph. 
        # Check current state for debugging
        cutoff_state = graph.get_state(config)
        logger.debug(f"Current State Check: {cutoff_state.values.keys() if cutoff_state else 'None'}")
        
        # If state is empty (server restart), we might want to ensure 'assistant' path is safe
        # But our router handles missing 'disease' by going to assistant.
        
        events = graph.invoke({"messages": [user_msg]}, config=config)
        logger.debug(f"Graph Events Keys: {events.keys()}")
        
        if "messages" in events and len(events["messages"]) > 0:
            response_msg = events["messages"][-1]
            logger.info(f"AI Reply: {response_msg.content[:50]}...")
            return ChatResponse(reply=response_msg.content)
        else:
            logger.warning("Graph did not return messages.")
            return ChatResponse(reply="Error: No response from AI agent.")
            
    except Exception as e:
        logger.error(f"Chat Endpoint Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return ChatResponse(reply=f"System Error: {str(e)}")

if __name__ == "__main__":
    is_dev = os.getenv("ENV", "development").lower() == "development"
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=is_dev)
