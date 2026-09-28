import os
import redis
import numpy as np
import json
import time
import typing as t
import requests
import logging
from persistant_memory.loading_and_saving_chat import get_top_k_queries
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

INDEX_NAME = "idx:semantic"
KEY_PREFIX = "semantic:"
EMBED_FIELD = "embedding"
DIM = int(os.getenv("DIM", 3072))
THRESHOLD = float(os.getenv("THRESHOLD", 0.98))
DEFAULT_K = 3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic-cache")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

def _init_redis_client():
    hosts = [REDIS_HOST, "localhost", "127.0.0.1"]
    for host in hosts:
        try:
            client = redis.Redis(host=host, port=REDIS_PORT, decode_responses=False, socket_timeout=2.0)
            client.ping()
            return client
        except Exception:
            continue
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False, socket_timeout=2.0)

r = _init_redis_client()




# -------------------------
# Helpers
# -------------------------
def to_float32(arr: np.ndarray) -> np.ndarray:
    """Return a float32 view/copy of arr avoiding copy if possible."""
    return np.asarray(arr, dtype=np.float32, order="C")

def normalize_inplace(arr: np.ndarray) -> np.ndarray:
    """Normalize vector in-place and return it (float32). If zero-vector, returns arr unchanged."""
    arr = to_float32(arr)
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        return arr
    arr /= norm
    return arr

def float32_array_to_bytes(arr: np.ndarray) -> bytes:
    """Convert float32 numpy array to raw bytes for Redis storage."""
    arr = to_float32(arr)
    # ensure C-order contiguous
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr.tobytes()

def bytes_to_float32_array(b: bytes) -> np.ndarray:
    """Convert raw bytes back to float32 numpy array (view, no copy)."""
    return np.frombuffer(b, dtype=np.float32)

# -------------------------
# Cosine / similarity helpers (assume normalized vectors)
# -------------------------
def dot_similarity_normalized(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity faster assuming a and b are already L2-normalized float32 vectors.
    """
    a = to_float32(a)
    b = to_float32(b)
    # direct dot — both normalized
    return float(np.dot(a, b))



# -------------------------
# Index creation helper (run once)
# -------------------------
def create_index_if_not_exists(dim: int = DIM):
    """Create RediSearch HNSW index configured for VECTOR FLOAT32 COSINE."""
    try:
        try:
            r.execute_command("FT.INFO", INDEX_NAME)
            logger.info("Index already exists: %s", INDEX_NAME)
            return
        except redis.exceptions.ResponseError:
            logger.info("Creating index: %s", INDEX_NAME)

        cmd = [
            "FT.CREATE", INDEX_NAME,
            "ON", "HASH",
            "PREFIX", "1", KEY_PREFIX,

            "SCHEMA",
                "query", "TEXT",
                "answer", "TEXT",

                # Vector field (all params inside schema)
                EMBED_FIELD, "VECTOR", "HNSW", "10",
                    "TYPE", "FLOAT32",
                    "DIM", str(dim),
                    "DISTANCE_METRIC", "COSINE",
                    "M", "16",
                    "EF_CONSTRUCTION", "200"
        ]

        res = r.execute_command(*cmd)
        logger.info("Index created: %s", res)
    except Exception as exc:
        logger.warning(f"⚠️ Redis semantic cache index creation skipped: {exc}")



# -------------------------
# Upsert a cached Q/A (normalize embedding before storing)
# -------------------------
import json
import numpy as np

def upsert_cache_item(id: str, query_text: str, answer_text: str, embedding: np.ndarray):
    key = f"{KEY_PREFIX}{id}"

    
    # If answer_text is a string, it means it's likely already JSON from SQLite.
    # We parse it so that json.dumps below produces a CLEAN string.
    if isinstance(answer_text, str):
        try:
            answer_text = json.loads(answer_text)
        except json.JSONDecodeError:
            pass
    

    emb = normalize_inplace(np.array(embedding, dtype=np.float32))
    emb_bytes = float32_array_to_bytes(emb)

    mapping = {
        "query": query_text.encode("utf-8"),
        "answer": json.dumps(answer_text).encode("utf-8"), 
        EMBED_FIELD: emb_bytes
    }
    r.hset(key, mapping=mapping)
    logger.debug("Upserted cache key=%s", key)






def _parse_search_response(res: list) -> t.List[t.Tuple[str, dict]]:
    """
    Parse the dialect-2 FT.SEARCH response to a list of (key, field_map) entries.
    Each field_map contains raw bytes as values.
    """
    if not res:
        return []
    total = res[0]
    if total == 0:
        return []

    entries: t.List[t.Tuple[str, dict]] = []
    i = 1
    for _ in range(total):
        raw_key = res[i]            # bytes
        key = raw_key.decode() if isinstance(raw_key, (bytes, bytearray)) else str(raw_key)
        i += 1
        fields = res[i]             # list of [field, value, ...] with bytes
        i += 1
        it = iter(fields)
        field_map: dict = {}
        for f in it:
            # f is field name (bytes)
            name = f.decode() if isinstance(f, (bytes, bytearray)) else str(f)
            val = next(it)
            field_map[name] = val
        entries.append((key, field_map))
    print("_parse_search_response completed")
    return entries



def semantic_lookup(query_text:str,query_vector:list, k: int = DEFAULT_K, threshold: float = THRESHOLD):

    q_emb = np.array(query_vector, dtype=np.float32)
    q_emb = normalize_inplace(q_emb)
    q_bytes = float32_array_to_bytes(q_emb)

    knn_clause = f"*=>[KNN {k} @{EMBED_FIELD} $vec AS score]"

    res = r.execute_command(
        "FT.SEARCH", INDEX_NAME,
        knn_clause,
        "PARAMS", "2", "vec", q_bytes,
        "SORTBY", "score",
        "LIMIT", "0", str(k),
        "DIALECT", "2",
        "RETURN", "3", "answer", "query", "score"
    )

    # print("RAW FT.SEARCH RESPONSE:", res)

    hits = _parse_search_response(res)
    if not hits:
        print("[CACHE] No hits")
        return None, 0.0, None

    best_score = -1.0
    best_answer = None
    best_key = None

    for key, field_map in hits:
        score_val = float(field_map["score"].decode())
        similarity = 1.0 - score_val   # COSINE distance → similarity

        answer = json.loads(field_map["answer"].decode())

        print(f"[CACHE] key={key}, similarity={similarity:.4f}")

        if similarity > best_score:
            best_score = similarity
            best_answer = answer
            best_key = key

    print(f"[CACHE] BEST similarity={best_score:.4f}")

    if best_score >= threshold:
        return best_answer, best_score, best_key

    return None, best_score, best_key






import typing as t
def cache_rag(query_text: str,query_vector: list,k: int = DEFAULT_K,):

    # -----------------------------
    # 1. Semantic Cache Lookup
    # -----------------------------
    print("answer got before cached")
    cached_answer, score, cache_key = semantic_lookup(query_text, query_vector,k=k)
    print("answer got after cached")

    # print("cached answer: ",cached_answer)
    # print("cached answer: ",score)
    # print("cached answer: ",cache_key)

    cache_result = None
    # query_emb = embd_model(query=query_text)

    print("query in cached len: ",len(query_vector))

    if cached_answer is not None:
        # Update existing cache item (refresh)
        # upsert_cache_item(
        #     cache_key.replace(KEY_PREFIX, ""),  # existing ID
        #     query_text,
        #     cached_answer,
        #     query_vector,
        # )

        cache_result = {
            "answer": cached_answer,
            "similarity": score,
            "cache_key": cache_key,
            "source": "semantic-cache",
        }
    # print("cached rag")
    # print("cached_result: ",cache_result)
    return {
        "cache": cache_result
    }




# -------------------------
# Helper example id generator
# -------------------------
def simple_id_generator() -> t.Callable[[], str]:
    counter = {"v": int(time.time() * 1000) % 1000000}
    def gen():
        counter["v"] += 1
        return str(counter["v"])
    return gen







import typing as t

def upsert_rag_response(
    rag_answer,
    query_text: str,
    query_vector: list,
    id_generator: t.Callable[[], str],
):
    
    rag_answer = rag_answer

    # -----------------------------
    # 3. Always Upsert RAG Result
    # -----------------------------
    # query_emb = embd_model(query=query_text)
    new_id = id_generator()
    upsert_cache_item(
        new_id,
        query_text,
        rag_answer,
        np.array(query_vector)
    )

    print("upsert rag reponse")

    rag_result = {
        "answer_test": rag_answer
    }

def clear_redis_cache():
    r.flushdb()  # or delete by prefix if shared



def refresh_redis_from_sqlite(limit=100):
    print(f"🔥 Refreshing Redis hot cache (top {limit})")
    clear_redis_cache()
    
    top_rows = get_top_k_queries(limit)
    id_gen = simple_id_generator()
    for query, answer,hits in top_rows:
        emb = embedding_model.embed_query(query)

        upsert_rag_response(
            json.loads(answer),
            query,
            np.array(emb) ,
            id_gen
        )

        # upsert_cache_item(
        #     id=id_gen, 
        #     query_text=query, 
        #     answer_text=json.loads(answer), 
        #     embedding=np.array(emb)
        #     )

    print("✅ Redis cache refreshed")