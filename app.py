
import streamlit as st
from PIL import Image, ExifTags
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import hashlib
import io
from datetime import datetime

# ==========================================
# VERISCOPE - DIGITAL CONTENT VERIFICATION
# ==========================================

MODEL_NAME = "bendavidsteel/openfake-swinv2-base"

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

.small {
    color: #9db1cc;
    font-size: 13px;
}

div[data-testid="stMetric"] {
    background: #14243b;
    padding: 16px;
    border: 1px solid #29415f;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# MODEL LOADING
# ==========================================

@st.cache_resource(show_spinner=False)
def load_detector():

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return processor, model


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
    "Analyse an image with an AI-detection model, inspect its "
    "metadata and generate a digital fingerprint."
)


# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded:

    image_bytes = uploaded.getvalue()

    fingerprint = hashlib.sha256(
        image_bytes
    ).hexdigest()

    # ======================================
    # READ IMAGE
    # ======================================

    try:

        original_image = Image.open(
            io.BytesIO(image_bytes)
        )

        original_image.load()

        image_format = original_image.format or "Unknown"

        image_width, image_height = original_image.size

        color_mode = original_image.mode

        image = original_image.convert("RGB")

    except Exception:

        st.error("Unable to read this image.")

        st.stop()


    # ======================================
    # RESET OLD RESULTS FOR NEW IMAGE
    # ======================================

    if st.session_state.get("current_fingerprint") != fingerprint:

        st.session_state["current_fingerprint"] = fingerprint

        st.session_state.pop("detector_result", None)

        st.session_state.pop("detector_error", None)


    # ======================================
    # FILE INFORMATION
    # ======================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

        st.caption(uploaded.name)


    with col2:

        st.subheader("File Information")

        st.write("**Format:**", image_format)

        st.write(
            "**Dimensions:**",
            f"{image_width} × {image_height}"
        )

        st.write("**Color mode:**", color_mode)

        st.write(
            "**File size:**",
            f"{len(image_bytes):,} bytes"
        )

        st.write("**SHA-256 Fingerprint**")

        st.code(fingerprint)


    # ======================================
    # METADATA INSPECTION
    # ======================================

    st.divider()

    st.subheader("📋 Metadata Inspection")

    exif_data = {}

    try:

        exif_data = original_image.getexif()

        if exif_data:

            metadata = {}

            for key, value in exif_data.items():

                name = ExifTags.TAGS.get(
                    key,
                    str(key)
                )

                metadata[name] = str(value)[:300]

            st.json(metadata)

        else:

            st.warning(
                "No EXIF metadata found. This does not prove "
                "that the image is AI-generated."
            )

    except Exception:

        exif_data = {}

        st.warning(
            "Metadata could not be read."
        )


    # ======================================
    # AI IMAGE DETECTION
    # ======================================

    st.divider()

    st.subheader("🤖 AI Image Detection")

    st.markdown("### OpenFake SwinV2")

    st.caption(
        "This research model classifies images as real or "
        "AI-generated. Its model card specifies a temperature "
        "scaling factor of 2."
    )

    if st.button(
        "Analyse Image",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Loading OpenFake and analysing image. "
                "The first model download may take several minutes..."
            ):

                processor, model = load_detector()

                inputs = processor(
                    images=image,
                    return_tensors="pt"
                )

                with torch.no_grad():

                    logits = model(**inputs).logits

                probabilities = torch.softmax(
                    logits / 2.0,
                    dim=-1
                )[0]

                # Model label order:
                # Index 0 = real
                # Index 1 = fake / AI-generated

                ai_score = float(
                    probabilities[1].item()
                )

                human_score = float(
                    probabilities[0].item()
                )

                st.session_state["detector_result"] = {
                    "ai": ai_score,
                    "human": human_score
                }

                st.session_state.pop(
                    "detector_error",
                    None
                )

        except Exception as error:

            st.session_state["detector_error"] = str(error)


    # ======================================
    # DISPLAY DETECTION RESULTS
    # ======================================

    if "detector_result" in st.session_state:

        result = st.session_state["detector_result"]

        col1, col2 = st.columns(2)

        col1.metric(
            "AI-generated",
            f"{result['ai'] * 100:.2f}%"
        )

        col2.metric(
            "Human-created",
            f"{result['human'] * 100:.2f}%"
        )

        if result["ai"] >= 0.5:

            st.warning(
                "The model classifies this image as more likely "
                "AI-generated."
            )

        else:

            st.success(
                "The model classifies this image as more likely "
                "real / human-created."
            )

        st.caption(
            "This is an automated model estimate, not proof of "
            "authenticity. The model can make mistakes."
        )


    elif "detector_error" in st.session_state:

        st.error(
            "The detector could not analyse this image: "
            + st.session_state["detector_error"][:500]
        )

    else:

        st.info(
            "Select Analyse Image to run the AI detector."
        )


    # ======================================
    # VERIFICATION REPORT
    # ======================================

    st.divider()

    st.subheader("📄 Verification Report")

    result = st.session_state.get("detector_result")

    if result:

        detector_text = (
            f"AI-generated score: "
            f"{result['ai'] * 100:.2f}%\n"
            f"Human-created score: "
            f"{result['human'] * 100:.2f}%"
        )

    else:

        detector_text = "Not completed"


    report = f"""
VERISCOPE - DIGITAL CONTENT VERIFICATION REPORT

Date: {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}

FILE INFORMATION
Name: {uploaded.name}
Format: {image_format}
Dimensions: {image_width} x {image_height}
Color mode: {color_mode}
Size: {len(image_bytes)} bytes

SHA-256 FINGERPRINT
{fingerprint}

METADATA
EXIF present: {"Yes" if exif_data else "No"}

AI IMAGE DETECTION
Model: {MODEL_NAME}
{detector_text}

DISCLAIMER
The model output is an automated estimate, not proof of authenticity.
The model can make mistakes. Additional evidence and human review
may be needed before deciding whether an image is authentic.

OpenFake model licence: CC-BY-NC-4.0 (non-commercial use).
"""


    st.download_button(
        "⬇ Download Verification Report",
        data=report,
        file_name="VERISCOPE_Report.txt",
        mime="text/plain",
        use_container_width=True
    )


# ==========================================
# WELCOME SCREEN
# ==========================================

else:

    with st.container(border=True):

        st.subheader("Welcome to VERISCOPE")

        st.write(
            "Upload an image to begin digital verification."
        )

        st.caption(
            "AI Image Detection • Metadata Inspection • "
            "SHA-256 Fingerprinting • Verification Reports"
        )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "VERISCOPE | Deepfake Detection & "
    "Digital Content Authentication Prototype"
)
