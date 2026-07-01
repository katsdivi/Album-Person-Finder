"""
One-time (and re-runnable) backfill: walks the Drive folder tree, detects
faces in every image, and stores embeddings in Postgres.

Safe to re-run any time (e.g. nightly via cron) -- already-indexed photos
are skipped, so it only processes files added since the last run.

GPU usage: downloads run in parallel threads to keep the GPU fed; face
inference runs on the main thread so the GPU is never contended.

Usage:
    python backfill.py
    python backfill.py --folder-id 1My4hb0f8XQnmKK3S3BMLS5EF3msnPGYW
    python backfill.py --download-workers 16   # more parallel Drive downloads
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

from drive_client import get_drive_service, walk_images, download_file_bytes
from face_index import extract_faces
import db

load_dotenv()


def run(folder_id: str, download_workers: int = 8):
    service = get_drive_service()
    print(f"Walking Drive folder {folder_id} ...")

    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM photos")
        already_seen = {row[0] for row in cur.fetchall()}
        cur.close()

    images = list(walk_images(service, folder_id))
    print(f"Found {len(images)} image files in Drive.")

    new_images = [img for img in images if img["id"] not in already_seen]
    print(f"{len(new_images)} are new since the last run.")
    if not new_images:
        print("Nothing to do.")
        return

    print(f"Using {download_workers} download threads + GPU inference.")

    indexed, skipped_errors = 0, 0
    with db.get_conn() as conn:
        cur = conn.cursor()

        def _download(img):
            return img, download_file_bytes(service, img["id"])

        with ThreadPoolExecutor(max_workers=download_workers) as pool:
            futures = {pool.submit(_download, img): img for img in new_images}
            for future in tqdm(as_completed(futures), total=len(new_images), desc="Indexing"):
                try:
                    img, content = future.result()
                    faces = extract_faces(content)

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
                except Exception as e:
                    skipped_errors += 1
                    tqdm.write(f"  ! Failed on {futures[future]['name']}: {e}")

        conn.commit()
        cur.close()

    print(f"Done. Indexed {indexed} new photos, {skipped_errors} errors/skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder-id",
        default=os.environ.get("DRIVE_ROOT_FOLDER_ID"),
        help="Drive folder ID to start walking from",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Parallel threads for Drive downloads (default: 8)",
    )
    args = parser.parse_args()
    if not args.folder_id:
        raise SystemExit("Set DRIVE_ROOT_FOLDER_ID in .env or pass --folder-id")

    db.init_schema()
    run(args.folder_id, args.download_workers)