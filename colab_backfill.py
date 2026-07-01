"""
A100-optimized backfill.

Pipeline:
  64 download threads  →  result_queue
  4 GPU inference threads  →  db_queue      (each has its own FaceAnalysis instance;
                                              ONNX Runtime releases the GIL during
                                              inference so they truly run in parallel)
  1 DB writer thread  →  Postgres           (GPU workers never stall on DB I/O)

Run:
    python colab_backfill.py
    python colab_backfill.py --gpu-workers 8   # tune up if VRAM allows
"""
import argparse
import os
import queue
import threading

import cv2
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

MAX_DIMENSION = 1600
DOWNLOAD_THREADS = 64
GPU_WORKERS = 4
DB_COMMIT_EVERY = 200


def _resize_if_huge(cv_img):
    h, w = cv_img.shape[:2]
    longest = max(h, w)
    if longest <= MAX_DIMENSION:
        return cv_img
    scale = MAX_DIMENSION / longest
    return cv2.resize(cv_img, (int(w * scale), int(h * scale)))


def _downloader_worker(service, work_queue, result_queue, done_counter, done_lock, gpu_workers):
    from drive_client import download_file_bytes

    while True:
        img = work_queue.get()
        if img is None:
            with done_lock:
                done_counter[0] += 1
                if done_counter[0] == DOWNLOAD_THREADS:
                    # Last downloader out — send one stop signal per GPU worker
                    for _ in range(gpu_workers):
                        result_queue.put(None)
            return
        try:
            content = download_file_bytes(service, img["id"])
            arr = np.frombuffer(content, dtype=np.uint8)
            cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if cv_img is None:
                result_queue.put({"ok": False, "img": img, "error": "could not decode"})
                continue
            cv_img = _resize_if_huge(cv_img)
            result_queue.put({"ok": True, "img": img, "cv_img": cv_img})
        except Exception as e:
            result_queue.put({"ok": False, "img": img, "error": str(e)})


def _gpu_worker(result_queue, db_queue):
    from insightface.app import FaceAnalysis

    face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    while True:
        item = result_queue.get()
        if item is None:
            db_queue.put(None)
            return

        img = item["img"]
        if not item["ok"]:
            db_queue.put({"ok": False, "img": img, "error": item["error"]})
            continue

        try:
            faces = face_app.get(item["cv_img"])
            face_data = [
                {"bbox": [float(x) for x in f.bbox], "embedding": f.normed_embedding}
                for f in faces
            ]
            db_queue.put({"ok": True, "img": img, "face_data": face_data})
        except Exception as e:
            db_queue.put({"ok": False, "img": img, "error": str(e)})


def _db_writer(db_queue, pbar, gpu_workers, results):
    import db

    finished = 0
    indexed = 0
    errors = 0

    with db.get_conn() as conn:
        cur = conn.cursor()
        while True:
            item = db_queue.get()
            if item is None:
                finished += 1
                if finished == gpu_workers:
                    break
                continue

            img = item["img"]
            if not item["ok"]:
                errors += 1
                tqdm.write(f"  ! {img['name']}: {item['error']}")
                pbar.update(1)
                continue

            db.insert_photo(
                cur,
                photo_id=img["id"],
                name=img["name"],
                folder_path=img.get("folder_path", ""),
                web_view_link=img.get("webViewLink"),
                thumbnail_link=img.get("thumbnailLink"),
                face_count=len(item["face_data"]),
            )
            for f in item["face_data"]:
                db.insert_face(cur, img["id"], f["bbox"], f["embedding"])

            indexed += 1
            pbar.update(1)
            if indexed % DB_COMMIT_EVERY == 0:
                conn.commit()

        conn.commit()
        cur.close()

    results["indexed"] = indexed
    results["errors"] = errors


def run(folder_id: str, num_threads: int, gpu_workers: int):
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

    print(f"Launching {gpu_workers} GPU workers + {num_threads} download threads...")

    work_queue: queue.Queue = queue.Queue()
    result_queue: queue.Queue = queue.Queue(maxsize=512)
    db_queue: queue.Queue = queue.Queue(maxsize=512)

    for img in new_images:
        work_queue.put(img)
    for _ in range(num_threads):
        work_queue.put(None)

    done_counter = [0]
    done_lock = threading.Lock()

    for _ in range(num_threads):
        threading.Thread(
            target=_downloader_worker,
            args=(get_drive_service(), work_queue, result_queue, done_counter, done_lock, gpu_workers),
            daemon=True,
        ).start()

    for _ in range(gpu_workers):
        threading.Thread(
            target=_gpu_worker,
            args=(result_queue, db_queue),
            daemon=True,
        ).start()

    results = {}
    with tqdm(total=len(new_images), desc=f"Indexing (A100 ×{gpu_workers})") as pbar:
        db_thread = threading.Thread(
            target=_db_writer,
            args=(db_queue, pbar, gpu_workers, results),
        )
        db_thread.start()
        db_thread.join()

    print(f"Done. Indexed {results.get('indexed', 0)} photos, {results.get('errors', 0)} errors/skipped.")


if __name__ == "__main__":
    import db

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-id", default=os.environ.get("DRIVE_ROOT_FOLDER_ID"))
    parser.add_argument("--threads", type=int, default=DOWNLOAD_THREADS)
    parser.add_argument("--gpu-workers", type=int, default=GPU_WORKERS)
    args = parser.parse_args()
    if not args.folder_id:
        raise SystemExit("Set DRIVE_ROOT_FOLDER_ID in .env or pass --folder-id")

    db.init_schema()
    run(args.folder_id, args.threads, args.gpu_workers)
