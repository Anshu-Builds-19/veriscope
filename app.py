
import streamlit as st
from PIL import Image, ExifTags
from transformers import pipeline
import hashlib
import io
from datetime import datetime

# ==========================================
# VERISCOPE - DIGITAL CONTENT VERIFICATION
# ==========================================

MODEL_ID = "dima806/ai_vs_human_generated_image_detection"

st.set_page_config(
    page_title="VERISCOPE | Digital Verification",
    page_icon="🔎",
    layout="wide"
)

# ==========================================
# DESIGN
# ==========================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f, #101d32);
    color: #edf4ff;
}
h1, h2, h3 {
    color: #eaf2ff;
}
.subtitle {
    color: #9db1cc;
    font-size: 17px;
}
.panel {
    background: #14243b;
    padding: 22px;
    border: 1px solid #29415f;
    border-radius: 14px;
    margin-bottom: 15px;
}
.small {
    color: #9db1cc;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD AI MODEL
# ==========================================

@st.cache_resource(show_spinner=False)
def load_detector():
    return pipeline(
        "image-classification",
        model=MODEL_ID,
        device=-1
    )

# ==========================================
# HEADER
# ==========================================

st.markdown("# 🔎 VERISCOPE")

st.markdown(
    '<p class="subtitle">Digital Content Verification & '
    'AI Image Detection Platform</p>',
    unsafe_allow_html=True
)

st.divider()

st.info(
    "Upload an image to inspect its metadata, calculate "
    "its SHA-256 fingerprint and estimate whether it is AI-generated."
)

# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG"
)

if uploaded:

    image_bytes = uploaded.getvalue()

    # Calculate fingerprint
    fingerprint = hashlib.sha256(image_bytes).hexdigest()

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        image = image.convert("RGB")

    except Exception:
        st.error("Unable to read this image. Please upload a valid image.")
        st.stop()

    # Clear old results when a different image is uploaded
    if st.session_state.get("current_fingerprint") != fingerprint:
        st.session_state["current_fingerprint"] = fingerprint
        st.session_state.pop("ai_results", None)
        st.session_state.pop("ai_error", None)

    # ======================================
    # IMAGE AND FILE INFORMATION
    # ======================================

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)
        st.caption(f"File: {uploaded.name}")

    with col2:
        st.subheader("File Information")

        st.write("**Format:**", uploaded.type or "Unknown")
        st.write(
            "**Dimensions:**",
            f"{image.width} × {image.height}"
        )
        st.write("**Color mode:**", image.mode)
        st.write(
            "**File size:**",
            f"{len(image_bytes):,} bytes"
        )

        st.write("**SHA-256 Fingerprint**")
        st.code(fingerprint, language=None)

    # ======================================
    # METADATA INSPECTION
    # ======================================

    st.divider()
    st.subheader("📋 Metadata Inspection")

    try:
        original_image = Image.open(io.BytesIO(image_bytes))
        exif_data = original_image.getexif()

        if exif_data:
            metadata = {}

            for key, value in exif_data.items():
                name = ExifTags.TAGS.get(key, str(key))
                metadata[name] = str(value)[:300]

            st.json(metadata)

        else:
            st.warning(
                "No EXIF metadata found. This does not prove "
                "that an image is AI-generated or manipulated."
            )

    except Exception:
        exif_data = {}
        st.warning("Metadata could not be read.")

    # ======================================
    # AI IMAGE DETECTION
    # ======================================

    st.divider()
    st.subheader("🤖 AI Image Detection")

    st.caption(
        "Model: dima806/ai_vs_human_generated_image_detection"
    )

    if st.button(
        "🔍 Analyse Image",
        type="primary",
        use_container_width=True
    ):

        try:
            with st.spinner(
                "Loading AI model and analysing image. "
                "The first run may take several minutes..."
            ):

                detector = load_detector()
                results = detector(image)

                # Identify exact model labels
                ai_result = next(
                    (
                        item for item in results
                        if item["label"].strip().lower()
                        == "ai-generated"
                    ),
                    None
                )

                human_result = next(
                    (
                        item for item in results
                        if item["label"].strip().lower()
                        == "human"
                    ),
                    None
                )

                if ai_result is None or human_result is None:
                    raise ValueError(
                        "Unexpected model labels: "
                        + str(results)
                    )

                st.session_state["ai_results"] = {
                    "ai_score": ai_result["score"],
                    "human_score": human_result["score"],
                    "all_results": results
                }

                st.session_state.pop("ai_error", None)

        except Exception as error:
            st.session_state["ai_error"] = str(error)
            st.session_state.pop("ai_results", None)

    # ======================================
    # DISPLAY AI RESULTS
    # ======================================

    if "ai_results" in st.session_state:

        result = st.session_state["ai_results"]

        ai_score = result["ai_score"]
        human_score = result["human_score"]

        st.markdown("### Analysis Results")

        score_col1, score_col2 = st.columns(2)

        with score_col1:
            st.metric(
                "AI-generated probability",
                f"{ai_score * 100:.2f}%"
            )

        with score_col2:
            st.metric(
                "Human-created probability",
                f"{human_score * 100:.2f}%"
            )

        st.write("AI-generated score")
        st.progress(float(ai_score))

        if ai_score >= 0.5:
            st.warning(
                "The model indicates stronger AI-generated signals."
            )
        else:
            st.success(
                "The model indicates stronger human-created signals."
            )

        st.markdown("### Model Output")

        for item in result["all_results"]:
            st.write(
                f"**{item['label']}** — "
                f"{item['score'] * 100:.2f}%"
            )

        st.caption(
            "These are model confidence scores, not verified "
            "probabilities of authenticity. Results can be incorrect, "
            "especially for edited, compressed or unfamiliar images."
        )

    elif "ai_error" in st.session_state:

        st.error("AI analysis could not be completed.")

        st.caption(
            "Technical error: " + st.session_state["ai_error"][:500]
        )

    # ======================================
    # VERIFICATION REPORT
    # ======================================

    st.divider()
    st.subheader("📄 Verification Report")

    if "ai_results" in st.session_state:

        result = st.session_state["ai_results"]

        ai_report = (
            f"AI-generated score: "
            f"{result['ai_score'] * 100:.2f}%\n"
            f"Human-created score: "
            f"{result['human_score'] * 100:.2f}%\n"
            f"Model: {MODEL_ID}\n"
        )

    else:
        ai_report = "AI analysis: Not completed\n"

    report = f"""
VERISCOPE - DIGITAL CONTENT VERIFICATION REPORT

Date: {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}

FILE INFORMATION
File name: {uploaded.name}
Format: {uploaded.type}
Dimensions: {image.width} x {image.height}
File size: {len(image_bytes)} bytes

SHA-256 FINGERPRINT
{fingerprint}

METADATA
EXIF metadata present: {"Yes" if exif_data else "No"}

AI IMAGE DETECTION
{ai_report}

DISCLAIMER
This report contains automated technical observations
and model estimates.

The AI model cannot guarantee whether an image is genuine,
AI-generated or manipulated.

Further human verification and additional evidence may
be required.
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
        AI Image Detection • Metadata Inspection • SHA-256
        Fingerprinting • Downloadable Reports
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "VERISCOPE | Deepfake Detection & "
    "Digital Content Authentication Prototype"
)
