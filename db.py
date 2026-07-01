"""
Postgres + pgvector storage layer.

Two tables:
  photos  -- one row per Drive image file we've processed (so re-running
             backfill is idempotent and skips already-indexed files)
  faces   -- one row per detected face, with a 512-dim embedding vector
             pointing back at its photo

Run `python db.py` once to create the schema on a fresh Supabase project.
"""
import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.environ["SUPABASE_DB_URL"]
EMBEDDING_DIM = 512  # InsightFace buffalo_l output size

SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS photos (
    id              TEXT PRIMARY KEY,      -- Google Drive file ID
    name            TEXT NOT NULL,
    folder_path     TEXT,                  -- human-readable path, e.g. "AKSHAY & SHREYA/Video"
    web_view_link   TEXT,
    thumbnail_link  TEXT,
    face_count      INTEGER DEFAULT 0,
    indexed_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS faces (
    id          SERIAL PRIMARY KEY,
    photo_id    TEXT REFERENCES photos(id) ON DELETE CASCADE,
    bbox        JSONB,
    embedding   VECTOR({EMBEDDING_DIM}) NOT NULL
);

-- Approximate nearest-neighbor index for fast cosine search at scale.
-- Safe to re-run; Postgres will skip if it already exists.
CREATE INDEX IF NOT EXISTS faces_embedding_idx
    ON faces USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


@contextmanager
def get_conn():
    conn = psycopg2.connect(DB_URL)
    register_vector(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema():
    # Step 1: Enable the pgvector extension using a plain connection.
    # register_vector() requires the type to already exist, so we cannot
    # use get_conn() here -- it would fail on a fresh database.
    raw = psycopg2.connect(DB_URL)
    try:
        raw.autocommit = True
        with raw.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    finally:
        raw.close()

    # Step 2: Now that the extension exists, use the normal path to create
    # the rest of the schema (tables + index).
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
    print("Schema ready.")


def is_photo_indexed(cur, photo_id: str) -> bool:
    cur.execute("SELECT 1 FROM photos WHERE id = %s", (photo_id,))
    return cur.fetchone() is not None


def insert_photo(cur, photo_id, name, folder_path, web_view_link, thumbnail_link, face_count):
    cur.execute(
        """
        INSERT INTO photos (id, name, folder_path, web_view_link, thumbnail_link, face_count)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
        """,
        (photo_id, name, folder_path, web_view_link, thumbnail_link, face_count),
    )


def insert_face(cur, photo_id, bbox, embedding):
    cur.execute(
        "INSERT INTO faces (photo_id, bbox, embedding) VALUES (%s, %s, %s)",
        (photo_id, psycopg2.extras.Json(bbox), embedding),
    )


def search_similar_faces(cur, query_embedding, limit=60, max_distance=0.5):
    """
    pgvector's <=> operator returns cosine *distance* (0 = identical).
    We join back to photos and de-duplicate so one photo with several
    matching faces only shows up once, keeping its best (smallest) distance.
    """
    cur.execute(
        """
        SELECT p.id, p.name, p.folder_path, p.web_view_link, p.thumbnail_link,
               MIN(f.embedding <=> %s) AS distance
        FROM faces f
        JOIN photos p ON p.id = f.photo_id
        WHERE f.embedding <=> %s < %s
        GROUP BY p.id, p.name, p.folder_path, p.web_view_link, p.thumbnail_link
        ORDER BY distance ASC
        LIMIT %s
        """,
        (query_embedding, query_embedding, max_distance, limit),
    )
    return cur.fetchall()


if __name__ == "__main__":
    init_schema()