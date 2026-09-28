import sqlite3
import sqlite_vec
import json
import time
import numpy as np
import struct
import os

def _sqlite_path_from_env():
    database_url = os.getenv("DATABASE_URL", "chat_history.db")
    if database_url and database_url.startswith("sqlite:///"):
        path = database_url.replace("sqlite:///", "", 1)
        db_path = os.path.abspath(path)
    else:
        db_path = os.path.abspath("chat_history.db")

    # If Docker created a directory by mistake on host volume mount, remove it
    if os.path.exists(db_path) and os.path.isdir(db_path):
        try:
            os.rmdir(db_path)
        except Exception:
            pass

    return db_path

DB_PATH = _sqlite_path_from_env()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    cursor = conn.cursor()

    # 1. Main table (Keep as is)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        confidence_score FLOAT,
        hit_count FLOAT DEFAULT 1,
        last_asked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(session_id, question)
    )
    """)

    # 2. Virtual Vector Table - Remove the 'id' column definition
    # We will use the built-in 'rowid' which is standard for virtual tables
    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_chat_history USING vec0(
        query_vector float[3072]
    )
    """)

    conn.commit()
    conn.close()

# Helper to convert list/numpy to bit-format for sqlite-vec
def serialize_vector(vector):
    return np.array(vector, dtype=np.float32).tobytes()

def save_chat_turn(session_id, question, answer_dict, query_vector,confidence_score):
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    cursor = conn.cursor()
    
    try:
        # 1. Upsert text data into the standard table
        # We use RETURNING id to get the rowid for the vector table
        cursor.execute("""
            INSERT INTO chat_history (session_id, timestamp, question, answer,confidence_score, hit_count,last_asked_at)
            VALUES (?, ?, ?, ?,?, 1, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id, question) DO UPDATE SET
                answer = excluded.answer,
                hit_count = chat_history.hit_count + 1,
                last_asked_at = CURRENT_TIMESTAMP
            RETURNING id
        """, (session_id, time.strftime("%Y-%m-%d %H:%M:%S"), question, json.dumps(answer_dict), confidence_score))
        
        result = cursor.fetchone()
        if not result:
            return
        row_id = result[0]

        
        # FIRST: Delete the existing vector if it exists to avoid UNIQUE/Primary Key constraints
        cursor.execute("DELETE FROM vec_chat_history WHERE rowid = ?", (row_id,))
        
        # SECOND: Insert the new vector
        cursor.execute("""
            INSERT INTO vec_chat_history(rowid, query_vector) 
            VALUES (?, ?)
        """, (row_id, serialize_vector(query_vector)))

        conn.commit()
    except Exception as e:
        print(f"❌ Error saving to SQLite: {e}")
        conn.rollback()
    finally:
        conn.close()



def search_history_semantic(query_vector, proximity_threshold=0.95, top_k=5):
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    cursor = conn.cursor()

    query_vec = serialize_vector(query_vector)

    try:
        # Search for top_k similar vectors, then sort those by confidence score
        cursor.execute("""
            SELECT 
                h.answer,
                vec_distance_cosine(v.query_vector, ?) AS distance,
                h.question,
                h.confidence_score,
                h.session_id
            FROM vec_chat_history v
            JOIN chat_history h ON v.rowid = h.id
            WHERE v.query_vector MATCH ?
              AND k = ?
            ORDER BY h.confidence_score DESC  -- Prioritize best quality answer
        """, (query_vec, query_vec, top_k))

        rows = cursor.fetchall()
    except Exception as e:
        print(f"⚠️ SQLite Search Error: {e}")
        rows = []
    finally:
        conn.close()

    if not rows:
        return None

    # Filter by similarity threshold
    valid_results = []
    for row in rows:
        answer_json, distance, original_question, conf_score, sess_id = row
        similarity = 1 - distance
        if similarity >= proximity_threshold:
            valid_results.append({
                "answer": json.loads(answer_json),
                "similarity": similarity,
                "question": original_question,
                "confidence": conf_score,
                "session_id": sess_id
            })

    # Return the first one (already sorted by confidence DESC in the SQL)
    return valid_results[0] if valid_results else None







def increment_hit_count(session_id: str, question: str):
    """Increment hit_count for a specific session + question."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE chat_history
        SET hit_count = hit_count + 1,
            last_asked_at = CURRENT_TIMESTAMP
        WHERE session_id = ?
          AND question = ?
    """, (session_id, question))

    conn.commit()
    conn.close()

def get_unique_query_count() -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM chat_history")
    count = cur.fetchone()[0]
    conn.close()
    return count

def get_top_k_queries(k: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT 
            q.question,
            q.answer,
            stats.total_hits
        FROM chat_history q
        JOIN (
            SELECT 
                question,
                SUM(hit_count) AS total_hits,
                MAX(last_asked_at) AS latest_time
            FROM chat_history
            GROUP BY question
        ) stats
        ON q.question = stats.question
        AND q.last_asked_at = stats.latest_time
        ORDER BY stats.total_hits DESC
        LIMIT ?
    """, (k,))

    rows = cur.fetchall()
    conn.close()
    return rows  # Returns list of (question, answer_json)



def get_full_conversation(session_id):
    """
    Returns the full conversation for a given session_id from SQLite.
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT timestamp, question, answer
        FROM chat_history
        WHERE session_id = ?
        ORDER BY id ASC
        """,
        (session_id,)
    )

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return {
            "session_id": session_id,
            "exists": False,
            "conversation": []
        }

    conversation = []
    for timestamp, question, answer_json in rows:
        conversation.append({
            "timestamp": timestamp,
            "question": question,
            "answer": json.loads(answer_json)
        })

    return {
        "session_id": session_id,
        "exists": True,
        "total_turns": len(conversation),
        "conversation": conversation
    }



def get_recent_conversation(session_id, last_n=5):
    """
    Returns last N conversation turns for a session_id from SQLite.
    """
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT timestamp, question, answer
        FROM chat_history
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (session_id, last_n)
    )

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return {
            "session_id": session_id,
            "exists": False,
            "conversation": []
        }

    # Reverse because we fetched DESC
    rows.reverse()

    conversation = []
    for timestamp, question, answer_json in rows:
        conversation.append({
            "timestamp": timestamp,
            "question": question,
            "answer": json.loads(answer_json)
        })

    return {
        "session_id": session_id,
        "exists": True,
        "returned_turns": len(conversation),
        "conversation": conversation
    }


def load_chat_history(session_id, last_n:int=None):
    init_db()
    data = (
        get_recent_conversation(session_id, last_n)
        if last_n else
        get_full_conversation(session_id)
    )

    if not data["exists"]:
        return {
            "session_id": session_id,
            "exists": False,
            "conversation": []
        }

    return {
        "session_id": session_id,
        "exists": True,
        "conversation": data["conversation"]
    }

# ---------- EXAMPLE ----------
# ans = load_chat_history("test_session_1",last_n=5)
# for i, turn in enumerate(ans.get("conversation"),1):
    # print(f"--- Turn {i} ---")
    # print("question:", turn)
    # print("response:", turn["answer"]["response"])
# print("ans", ans)
