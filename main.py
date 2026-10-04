from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from fastapi.responses import JSONResponse
import time
from src.step_3_llm_loaders import main
# from workflow_milvus import process_pdf_folder
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Header
from pathlib import Path
from urllib.parse import unquote,quote
from fastapi.responses import FileResponse

# from data_cleaning.markdown_cleaning import process_all_files
# from drop_collection import drop_collection_from_milvus

# Enterprise RAG Security, Monitoring, and Resilience Pipeline
from src.pipeline import (
    security_guard,
    RateLimitMiddleware,
    llm_circuit_breaker,
    milvus_circuit_breaker,
    CircuitBreakerOpenException,
    cost_manager,
    output_validator,
    metrics_registry,
    langsmith_tracer
)

BASE_DIR = os.getcwd()
app = FastAPI()

app.add_middleware(RateLimitMiddleware)

origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost,http://localhost:3000,http://localhost:5173",
    ).split(",")
    if origin.strip()
]
# Add CORS middleware to your app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # List of allowed origins
    allow_credentials=True,           # Allow cookies and auth headers
    allow_methods=["*"],              # Allow all HTTP methods
    allow_headers=["*"],              # Allow all headers
)



# Request model: user sends a question
class QuestionRequest(BaseModel):
    question: str
    answer_type: str
    session_id: str
# # Response model: we respond with JSON
class AnswerResponse(BaseModel):
    answer: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    return {"status": "ok"}

# @app.post("/query_old", response_model=AnswerResponse)
# def answer_question(request: QuestionRequest):
#     user_question = request.question
#     start_time = time.time()

#     # db_load = loading_milvus()
#     response_data = asyncio.run(main(query=user_question))
#     elapsed_time = time.time() - start_time

#     print(f"\nTotal time consumed: {elapsed_time:.2f} seconds")

#     return JSONResponse(response_data)



class FolderPathRequest(BaseModel):
    folder_path: str

# @app.post("/data_ingestion")
# def data_ingestion_into_milvus(request: FolderPathRequest):
#     """
#     Accepts a folder path, processes all files in it,
#     and inserts data into Milvus.
#     """
#     input_folder = os.getenv("markdown_input_folder")
#     output_folder = os.getenv("markdown_output_folder")

#     if not input_folder:
#         raise ValueError("❌ Environment variable 'markdown_input_folder' not set.")

#     if not output_folder:
#         raise ValueError("❌ Environment variable 'markdown_output_folder' not set.")

#     print(f"📥 Input Folder: {input_folder}")
#     print(f"📤 Output Folder: {output_folder}")

#     process_all_files(input_folder, output_folder)
#     print("Done!")

#     folder_path = request.folder_path

#     if not os.path.exists(folder_path):
#         return JSONResponse(
#             status_code=400,
#             content={"error": f"Folder path '{folder_path}' does not exist."}
#         )

#     try:
#         process_folder(root_folder=folder_path)
#         return JSONResponse(
#             content={"message": f"Data ingestion completed successfully for folder: {folder_path}"}
#         )
#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"error": f"Failed to ingest data: {str(e)}"}
#         )


# @app.get("/drop")
# def data_drop_from_milvus():
#     drop_collection_from_milvus()

#     return {"message": "collection dropped"}



# from fastapi import FastAPI
# from fastapi.responses import FileResponse
# from urllib.parse import unquote
# import os

# @app.get("/open-pdf")
# async def open_pdf(file_path: str):
#     decoded_path = unquote(file_path)
#     clean_path = os.path.normpath(decoded_path)

#     print("Trying to open:", clean_path)

#     if not os.path.exists(clean_path):
#         return {"error": f"File not found: {clean_path}"}

#     return FileResponse(
#         clean_path,
#         media_type="application/pdf",
#         headers={
#             "Content-Disposition": f"inline; filename={os.path.basename(clean_path)}"
#         }
#     )






import typing as t
# from caching_history.caching.redis_semantic_cache import cache_rag,upsert_rag_response,simple_id_generator,create_index_if_not_exists,refresh_redis_from_sqlite

from fastapi import FastAPI, Query
DEFAULT_K = 1
# from caching_history.caching.redis_semantic_cache import create_index_if_not_exists
DIM = os.getenv("DIM",1024)   
# idgen = simple_id_generator() 
from fastapi import FastAPI, Depends ,HTTPException
from sqlalchemy.orm import Session
from caching_history.chat_history.database import Base, engine, get_db
from caching_history.chat_history.models import ChatSession, ChatMessage
from caching_history.chat_history.schemas import SessionCreate, ChatMessageCreate
import json
from caching_history.chat_history.helpers import get_all_chats_by_session , get_all_sessions_with_first_message
# from src.step_8_session_history import get_recent_history
# from step_5_prompt import build_conversation_prompt
from sqlalchemy.orm import Session
from multilingual_pipeline.language_detection import detect_language
from multilingual_pipeline.conversion import output_converison,translation
from milvus_database.milvus_loading import loading_milvus
from persistant_memory.loading_and_saving_chat import init_db

import sqlite3
DB_PATH = "chat_history.db"
DB_PATH = os.path.join(BASE_DIR,DB_PATH)

# Create tables
Base.metadata.create_all(bind=engine)


@app.post("/chat/session")
def create_session(payload: SessionCreate, db: Session = Depends(get_db)):
    session = ChatSession(title=payload.title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return {"session_id": session.id, "title": session.title}


import uuid
# from fastapi import APIRouter

# app = APIRouter()

# def generate_session_id():
#     return str(uuid.uuid4())

# @app.post("/chat/session")
# def create_chat_session():
#     session_id = generate_session_id()

#     return {
#         "session_id": session_id,
#         "message": "New chat session created"
#     }


# ---------------------
# Save user or bot message
# ---------------------
@app.post("/chat/message")
def store_chat_message(payload: ChatMessageCreate, db: Session = Depends(get_db)):
    # Convert lists/dicts to JSON strings if present
    # print("payload: ",payload)
    msg = ChatMessage(
        session_id=payload.session_id,
        role=payload.role,
        response=payload.response,
        bold_words=json.dumps(payload.bold_words) if payload.bold_words else None,
        meta_data=json.dumps(payload.meta_data) if payload.meta_data else None,
        follow_up=payload.follow_up,
        table_data=json.dumps(payload.table_data) if payload.table_data else None,
        ucid=payload.ucid
    )

    db.add(msg)
    db.commit()
    db.refresh(msg)

    return {"status": "success", "message_id": msg.id}

# ---------------------
# Get all messages for a session
# ---------------------
# @app.get("/sessions/{session_id}/chats")
# def get_chats_by_session(session_id: str):
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()

#     cursor.execute("""
#         SELECT
#             id,
#             timestamp,
#             question,
#             answer,
#             confidence_score,
#             hit_count
#         FROM chat_history
#         WHERE session_id = ?
#         ORDER BY timestamp ASC
#     """, (session_id,))

#     rows = cursor.fetchall()
#     conn.close()

#     if not rows:
#         raise HTTPException(status_code=404, detail="No chats found for this session")

#     chats = []
#     for row in rows:
#         chat_id, ts, q, ans, conf, hits = row
#         chats.append({
#             "id": chat_id,
#             "timestamp": ts,
#             "question": q,
#             "answer": json.loads(ans),
#             "confidence_score": conf,
#             "hit_count": hits
#         })

#     return {
#         "session_id": session_id,
#         "total_messages": len(chats),
#         "chats": chats
#     }

# ---------------------
# Get all messages for a session
# ---------------------
@app.get("/sessions/{session_id}/chats")
def get_chats(session_id: str, db: Session = Depends(get_db)):
    chats = get_all_chats_by_session(db, session_id)

    if not chats:
        raise HTTPException(status_code=404, detail="No chats found for this session")

    return {
        "session_id": session_id,
        "total_messages": len(chats),
        "chats": chats
    }





# ---------------------
# Get  session list
# ---------------------

@app.get("/sessions")
def list_sessions(db: Session = Depends(get_db)):
    sessions = get_all_sessions_with_first_message(db)

    return {
        "total_sessions": len(sessions),
        "sessions": sessions
    }

# @app.get("/sessions")
# def list_sessions():
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()

#     cursor.execute("""
#         SELECT
#             session_id,
#             MIN(timestamp) AS first_time,
#             question
#         FROM chat_history
#         GROUP BY session_id
#         ORDER BY first_time DESC
#     """)

#     rows = cursor.fetchall()
#     conn.close()

#     sessions = []
#     for sess_id, first_time, first_question in rows:
#         sessions.append({
#             "session_id": sess_id,
#             "created_at": first_time,
#             "title": first_question[:80]  # frontend-friendly
#         })

#     return {
#         "total_sessions": len(sessions),
#         "sessions": sessions
#     }


#

@app.get("/metrics")
def get_metrics():
    """
    Exposes live system metrics: latency, throughput, token usage, costs, cache hit/miss, security.
    """
    return metrics_registry.get_metrics_snapshot()

db_load = loading_milvus() 

@app.post("/query", response_model=AnswerResponse)
async def answer_question(request: QuestionRequest, db: Session = Depends(get_db)):
    start_time = time.time()
    user_question = request.question
    answer_type = request.answer_type
    session_id = request.session_id

    # 1. Security Check: Input Sanitization
    sanitized_question = security_guard.sanitize_input(user_question)

    # 2. Security Check: Prompt Injection Defense
    is_injected, injection_pattern = security_guard.check_prompt_injection(sanitized_question)
    if is_injected:
        metrics_registry.record_security(blocked=True)
        metrics_registry.record_request(latency_ms=(time.time() - start_time) * 1000, is_error=True)
        raise HTTPException(
            status_code=400,
            detail=f"Security alert: Prompt injection or jailbreak pattern detected ('{injection_pattern}'). Request blocked."
        )

    # 3. Security Check: PII Masking on Input
    masked_query, input_pii_counts = security_guard.mask_pii(sanitized_question)
    total_pii_found = sum(input_pii_counts.values())
    if total_pii_found > 0:
        metrics_registry.record_security(pii_count=total_pii_found)

    # 4. Cost Governance: Token & Budget Check
    input_tokens = cost_manager.estimate_tokens(masked_query)
    routed_model = cost_manager.route_model(masked_query)

    try:
        # 5. Multilingual Translation
        detected_lang = detect_language(masked_query)
        translate_query = translation(detected_lang=detected_lang, user_query=masked_query)

        # 6. Execute RAG Main Logic
        response_data = await main(
            query=translate_query,
            db=db,
            session_id=session_id,
            answer_type=answer_type
        )

        response_payload = response_data[0] if isinstance(response_data, tuple) else response_data

        # Ensure dictionary response
        if not isinstance(response_payload, dict):
            response_payload = {"response": str(response_payload), "confidence_score": 0.5}

        raw_response_text = response_payload.get("response", "")

        # 7. Output Security: PII Masking on Output
        masked_response_text, output_pii_counts = security_guard.mask_pii(raw_response_text)
        response_payload["response"] = masked_response_text
        if sum(output_pii_counts.values()) > 0:
            metrics_registry.record_security(pii_count=sum(output_pii_counts.values()))

        # 8. Token & Cost Calculation
        output_tokens = cost_manager.estimate_tokens(masked_response_text)
        request_cost_usd = cost_manager.calculate_cost(routed_model, input_tokens, output_tokens)
        cost_manager.check_budget_and_record(request_cost_usd)

        # 9. Record Metrics & Telemetry
        elapsed_ms = (time.time() - start_time) * 1000
        metrics_registry.record_request(latency_ms=elapsed_ms, is_error=False)
        metrics_registry.record_tokens(input_tokens, output_tokens, request_cost_usd)

        # 10. LangSmith Trace Logging
        langsmith_tracer.log_trace_metadata(
            run_id=session_id,
            name="RAG_Query_Execution",
            metadata={
                "inputs": {"question": masked_query},
                "outputs": response_payload,
                "cost_usd": request_cost_usd,
                "latency_ms": elapsed_ms
            },
            tags=["rag", routed_model, "sanitized"]
        )

        # Save to DB
        db.add(ChatMessage(
            session_id=session_id,
            role="user",
            response=user_question,
        ))
        db.add(ChatMessage(
            session_id=session_id,
            role="assistant",
            response=response_payload.get("response", ""),
            bold_words=json.dumps(response_payload.get("bold_words", [])) if isinstance(response_payload.get("bold_words"), list) else None,
            meta_data=json.dumps(response_payload.get("meta_data", [])) if isinstance(response_payload.get("meta_data"), list) else None,
            follow_up=response_payload.get("follow_up"),
            table_data=json.dumps(response_payload.get("table_data", [])) if isinstance(response_payload.get("table_data"), list) else None,
            ucid=response_payload.get("ucid"),
        ))
        db.commit()

        print(f"\nQuery completed in {elapsed_ms:.2f} ms | Cost: ${request_cost_usd:.6f}")
        return JSONResponse(response_payload)

    except Exception as exc:
        elapsed_ms = (time.time() - start_time) * 1000
        metrics_registry.record_request(latency_ms=elapsed_ms, is_error=True)
        if isinstance(exc, HTTPException):
            raise exc
        print(f"Error processing query: {exc}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(exc)}")

