"""
Thin wrapper around InsightFace so the rest of the code doesn't need to know
about model names, providers, etc. CPU-only by default (ctx_id=-1) since this
is meant to run on a normal laptop for free, no GPU required.
"""
import cv2
import numpy as np
from insightface.app import FaceAnalysis

_app = None


def get_face_app(use_gpu: bool = True):
    global _app
    if _app is None:
        _app = FaceAnalysis(name="buffalo_l")
        ctx_id = 0 if use_gpu else -1
        _app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return _app


def extract_faces(image_bytes: bytes):
    """
    Returns a list of {"bbox": [x1,y1,x2,y2], "embedding": np.ndarray(512,)}
    for every face found in the image. Embeddings are L2-normalized so
    cosine similarity == dot product (pgvector handles this directly).
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    faces = get_face_app().get(img)
    results = []
    for face in faces:
        emb = face.normed_embedding  # already L2-normalized by insightface
        results.append({"bbox": [float(x) for x in face.bbox], "embedding": emb})
    return results


def extract_largest_face_embedding(image_bytes: bytes):
    """Convenience for the search UI: one reference photo -> one embedding
    (the largest face, in case the selfie has other people in the background)."""
    faces = extract_faces(image_bytes)
    if not faces:
        return None
    def area(f):
        x1, y1, x2, y2 = f["bbox"]
        return (x2 - x1) * (y2 - y1)
    return max(faces, key=area)["embedding"]