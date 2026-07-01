"""
One-time (and re-runnable) backfill: walks the Drive folder tree, detects
faces in every image, and stores embeddings in Postgres.

Safe to re-run any time (e.g. nightly via cron) -- already-indexed photos
are skipped, so it only processes files added since the last run.

Usage:
    python backfill.py
    python backfill.py --folder-id 1My4hb0f8XQnmKK3S3BMLS5EF3msnPGYW
"""
import argparse
import os

from dotenv import load_dotenv
from tqdm import tqdm

from drive_client import get_drive_service, walk_images, download_file_bytes
from face_index import extract_faces
import db

load_dotenv()


def run(folder_id: str):
    service = get_drive_service()
    print(f"Walking Drive folder {folder_id} ...")

    with db.get_conn() as conn:
        cur = conn.cursor()
        already_seen = set()
        cur.execute("SELECT id FROM photos")
        for (pid,) in cur.fetchall():
            already_seen.add(pid)
        cur.close()

    images = list(walk_images(service, folder_id))
    print(f"Found {len(images)} image files in Drive.")

    new_images = [img for img in images if img["id"] not in already_seen]
    print(f"{len(new_images)} are new since the last run.")

    indexed, skipped_errors = 0, 0
    with db.get_conn() as conn:
        cur = conn.cursor()
        for img in tqdm(new_images, desc="Indexing"):
            try:
                content = download_file_bytes(service, img["id"])
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
                    conn.commit()  # commit periodically so a crash doesn't lose all progress
            except Exception as e:
                skipped_errors += 1
                tqdm.write(f"  ! Failed on {img['name']} ({img['id']}): {e}")
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
    args = parser.parse_args()
    if not args.folder_id:
        raise SystemExit("Set DRIVE_ROOT_FOLDER_ID in .env or pass --folder-id")

    db.init_schema()
    run(args.folder_id)