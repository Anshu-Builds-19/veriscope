
import streamlit as st
from PIL import Image, ExifTags
import hashlib
import io
from datetime import datetime

st.set_page_config(
    page_title="VERISCOPE | Digital Verification",
    page_icon="🔎",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f, #101d32);
    color: #edf4ff;
}
h1, h2, h3 { color: #eaf2ff; }
.subtitle { color: #9db1cc; font-size: 17px; }
.panel {
    background: #14243b;
    padding: 22px;
    border: 1px solid #29415f;
    border-radius: 14px;
    margin-bottom: 15px;
}
.small { color: #9db1cc; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🔎 VERISCOPE")
st.markdown(
    '<p class="subtitle">Digital Content Verification & '
    'AI Image Detection Platform</p>',
    unsafe_allow_html=True
)

st.divider()

st.info(
    "Upload an image to inspect its file fingerprint, "
    "metadata and AI-generation indicators."
)

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG"
)

if uploaded:
    image_bytes = uploaded.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    image.load()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)
        st.caption(f"File: {uploaded.name}")

    with col2:
        st.subheader("File Information")

        st.write("**Format:**", image.format or "Unknown")
        st.write("**Dimensions:**", f"{image.width} × {image.height}")
        st.write("**Color mode:**", image.mode)
        st.write("**File size:**", f"{len(image_bytes):,} bytes")

        fingerprint = hashlib.sha256(image_bytes).hexdigest()

        st.write("**SHA-256 fingerprint**")
        st.code(fingerprint, language=None)

    st.divider()
    st.subheader("Metadata Inspection")

    exif_data = image.getexif()

    if exif_data:
        metadata = {}
        for key, value in exif_data.items():
            name = ExifTags.TAGS.get(key, str(key))
            metadata[name] = str(value)[:300]

        st.json(metadata)
    else:
        st.warning(
            "No EXIF metadata found. This does not prove "
            "that the image is AI-generated or manipulated."
        )

    st.divider()
    st.subheader("AI Image Detection")

    if st.button("🔍 Analyse Image", type="primary",
                 use_container_width=True):

        with st.spinner("Loading AI detection model..."):
            try:
                import torch
                from transformers import pipeline

                @st.cache_resource
                def load_detector():
                    return pipeline(
                        "image-classification",
                        model=(
                            "aaronkantrowitz/"
                            "ai-human-generated-image-detection"
                        ),
                        device=-1
                    )

                detector = load_detector()
                results = detector(image.convert("RGB"))

                ai_result = next(
                    (r for r in results
                     if "ai" in r["label"].lower()
                     or "fake" in r["label"].lower()
                     or "synthetic" in r["label"].lower()),
                    None
                )

                if ai_result is None:
                    human_result = next(
                        (r for r in results
                         if "human" in r["label"].lower()
                         or "real" in r["label"].lower()),
                        None
                    )
                    if human_result is not None:
                        ai_score = 1 - human_result["score"]
                    else:
                        raise ValueError(
                            "Could not identify the model's output labels."
                        )
                else:
                    ai_score = ai_result["score"]

                st.session_state["ai_score"] = ai_score
                st.session_state["model_results"] = results

            except Exception as error:
                st.session_state["model_error"] = str(error)

    if "ai_score" in st.session_state:
        score = st.session_state["ai_score"]

        st.metric("AI-generation score", f"{score * 100:.1f}%")
        st.progress(float(max(0, min(1, score))))

        if score >= 0.5:
            st.warning(
                "The model indicates stronger AI-generated signals."
            )
        else:
            st.success(
                "The model indicates stronger human-created signals."
            )

        st.caption(
            "This is a model estimate, not proof of authenticity. "
            "Results can be wrong, especially for edited, compressed "
            "or unfamiliar images."
        )

    elif "model_error" in st.session_state:
        st.error("The AI model could not complete the analysis.")
        st.caption(st.session_state["model_error"])

    st.divider()

    st.subheader("Verification Report")

    report = f"""
VERISCOPE - DIGITAL CONTENT REPORT

Date: {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}
File: {uploaded.name}
Format: {image.format}
Dimensions: {image.width} x {image.height}
File size: {len(image_bytes)} bytes

SHA-256:
{fingerprint}

EXIF metadata present: {"Yes" if exif_data else "No"}

AI detection:
{
    f"Model score: {st.session_state['ai_score'] * 100:.1f}%"
    if "ai_score" in st.session_state
    else "Not completed"
}

DISCLAIMER:
Automated detection is not proof of authenticity.
Human verification and additional evidence may be required.
"""

    st.download_button(
        "⬇ Download Verification Report",
        data=report,
        file_name="VERISCOPE_Report.txt",
        mime="text/plain",
        use_container_width=True
    )

else:
    st.markdown("""
    <div class="panel">
        <h3>Welcome to VERISCOPE</h3>
        <p>Upload an image to begin digital content verification.</p>
        <p class="small">
        Image analysis • Metadata inspection • SHA-256 fingerprint
        • AI detection • Downloadable reports
        </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption(
    "VERISCOPE | Prototype for Deepfake Detection "
    "& Digital Content Authentication"
)

