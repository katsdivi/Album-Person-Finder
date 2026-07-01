"""
Streamlit search app. Anyone in the family opens this in a browser, uploads
a photo of themselves (old or new -- any clear face works), and gets back
every matching photo from Drive with a link to open it.

Run locally:    streamlit run app.py
Or deploy free: push this repo to GitHub, deploy on share.streamlit.io,
                add the .env values as "Secrets" in the app settings.
"""
import os

import streamlit as st
from dotenv import load_dotenv

import db
from face_index import extract_largest_face_embedding

load_dotenv()

MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", 0.5))

st.set_page_config(page_title="Family Photo Face Search", page_icon="🔍", layout="wide")
st.title("🔍 Find yourself in the family photos")
st.caption(
    "Upload a clear photo of your face -- it doesn't need to be recent. "
    "An old photo works fine too, as long as your face is visible and not too small."
)

uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])

col_threshold, _ = st.columns([1, 3])
with col_threshold:
    threshold = st.slider(
        "Match strictness (lower = stricter, fewer but more confident matches)",
        min_value=0.2,
        max_value=0.8,
        value=MATCH_THRESHOLD,
        step=0.05,
    )

if uploaded is not None:
    image_bytes = uploaded.read()
    st.image(image_bytes, caption="Your reference photo", width=200)

    with st.spinner("Detecting your face..."):
        embedding = extract_largest_face_embedding(image_bytes)

    if embedding is None:
        st.error("Couldn't detect a clear face in that photo -- try a different one.")
    else:
        with st.spinner("Searching the family archive..."):
            with db.get_conn() as conn:
                cur = conn.cursor()
                results = db.search_similar_faces(
                    cur, embedding, limit=100, max_distance=threshold
                )
                cur.close()

        st.subheader(f"Found {len(results)} matching photo(s)")

        if not results:
            st.info(
                "No matches yet. Try loosening the strictness slider, or this person "
                "may not be indexed yet -- ask the admin to re-run the backfill."
            )

        cols = st.columns(4)
        for i, (photo_id, name, folder_path, web_view_link, thumbnail_link, distance) in enumerate(results):
            with cols[i % 4]:
                if thumbnail_link:
                    st.image(thumbnail_link, use_container_width=True)
                st.caption(f"{name}\n📁 {folder_path or 'root'}")
                if web_view_link:
                    st.link_button("Open in Drive", web_view_link, use_container_width=True)
else:
    st.info("Upload a photo above to start searching.")

st.divider()
st.caption(
    "Privacy note: this tool only stores numeric face descriptors and Drive links, "
    "not your uploaded photo. It's used in-memory for one search and then discarded."
)