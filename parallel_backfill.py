"""
CPU-only parallel backfill. Uses multiple CPU processes (each runs its own
Drive-download + face-detection pipeline).

NOTE: If you have a GPU, use backfill.py instead -- it runs GPU inference on
the main thread with parallel download threads and will be significantly faster.
This script is for CPU-only machines where spawning N worker processes across
cores is the best option.

Usage:
    python parallel_backfill.py
    python parallel_backfill.py --folder-id <ID>
    python parallel_backfill.py --workers 4   # override the auto core count
"""
import argparse
import os

# Stop each worker *process* from also spinning up multiple internal
# threads -- with N worker processes already saturating your cores, these
# math libraries fighting each other for the same cores actually slows
# things down. One thread per worker process is the sweet spot. Must be
# set before cv2/onnxruntime get imported anywhere.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import multiprocessing as mp
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Downscale huge photos before face detection. Faces are still plenty
# large to detect accurately at this resolution, but decoding/processing
# a 4000px photo vs a 1600px one is a big chunk of the per-photo cost.
MAX_DIMENSION = 1600

# Per-worker-process global state -- each worker loads its own copy of the
# face model and its own Drive login once, then reuses them for every
# photo it processes (instead of reloading per photo).
_worker_service = None
_worker_face_app = None


def _resize_if_huge(image_bytes):
    import cv2
    import numpy as np

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= MAX_DIMENSION:
        return image_bytes
    scale = MAX_DIMENSION / longest
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", resized)
    return buf.tobytes() if ok else image_bytes


def _worker_init():
    """Runs once when each worker process starts."""
    global _worker_service, _worker_face_app
    from drive_client import get_drive_service
    from face_index import get_face_app

    _worker_service = get_drive_service()
    _worker_face_app = get_face_app()


def _process_one(img):
    """Download + detect faces for a single Drive file. Runs inside a worker."""
    from drive_client import download_file_bytes
    import cv2
    import numpy as np

    try:
        content = download_file_bytes(_worker_service, img["id"])
        content = _resize_if_huge(content)

        arr = np.frombuffer(content, dtype=np.uint8)
        cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if cv_img is None:
            return {"ok": False, "img": img, "error": "could not decode image"}

        faces = _worker_face_app.get(cv_img)
        face_data = [
            {"bbox": [float(x) for x in f.bbox], "embedding": f.normed_embedding}
            for f in faces
        ]
        return {"ok": True, "img": img, "faces": face_data}
    except Exception as e:
        return {"ok": False, "img": img, "error": str(e)}


def run(folder_id: str, num_workers: int):
    from drive_client import get_drive_service, walk_images
    import db

    print(f"Listing images under folder {folder_id} ...")
    service = get_drive_service()
    images = list(walk_images(service, folder_id))
    print(f"Found {len(images)} image files in Drive.")

    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM photos")
        already_seen = {row[0] for row in cur.fetchall()}
        cur.close()

    new_images = [img for img in images if img["id"] not in already_seen]
    print(f"{len(new_images)} are new since the last run.")
    if not new_images:
        print("Nothing to do.")
        return

    print(f"Using {num_workers} parallel worker processes.")

    indexed, errors = 0, 0
    with db.get_conn() as conn:
        cur = conn.cursor()
        with mp.Pool(processes=num_workers, initializer=_worker_init) as pool:
            for result in tqdm(
                pool.imap_unordered(_process_one, new_images, chunksize=4),
                total=len(new_images),
                desc="Indexing",
            ):
                img = result["img"]
                if not result["ok"]:
                    errors += 1
                    tqdm.write(f"  ! Failed on {img['name']}: {result['error']}")
                    continue

                faces = result["faces"]
                db.insert_photo(
                    cur,
                    photo_id=img["id"],
                    name=img["name"],
                    folder_path=img.get("folder_path", ""),
                    web_view_link=img.get("webViewLink"),
                    thumbnail_link=img.get("thumbnailLink"),
                    face_count=len(faces),
                )
                for f in faces:
                    db.insert_face(cur, img["id"], f["bbox"], f["embedding"])

                indexed += 1
                if indexed % 50 == 0:
                    conn.commit()
        conn.commit()
        cur.close()

    print(f"Done. Indexed {indexed} new photos, {errors} errors/skipped.")


if __name__ == "__main__":
    import db

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-id", default=os.environ.get("DRIVE_ROOT_FOLDER_ID"))
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Number of parallel worker processes (default: CPU cores - 1)",
    )
    args = parser.parse_args()
    if not args.folder_id:
        raise SystemExit("Set DRIVE_ROOT_FOLDER_ID in .env or pass --folder-id")

    db.init_schema()
    run(args.folder_id, args.workers)