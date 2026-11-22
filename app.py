
import streamlit as st
from PIL import Image, ExifTags
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
import torch
import hashlib
import io
from datetime import datetime

# ==========================================
# VERISCOPE
# ==========================================

MODEL_ONE = "dima806/ai_vs_human_generated_image_detection"
MODEL_TWO = "bendavidsteel/openfake-swinv2-base"

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
# MODEL LOADING
# ==========================================

@st.cache_resource(show_spinner=False)
def load_detector_one():
    return pipeline(
        "image-classification",
        model=MODEL_ONE,
        device=-1
    )


@st.cache_resource(show_spinner=False)
def load_detector_two():
    processor = AutoImageProcessor.from_pretrained(MODEL_TWO)

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_TWO
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
    "Analyse images using two different AI detection models, "
    "inspect metadata and generate a digital fingerprint."
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

    fingerprint = hashlib.sha256(image_bytes).hexdigest()

    try:
        original_image = Image.open(io.BytesIO(image_bytes))
        original_image.load()

        image_format = original_image.format or "Unknown"
        image_width, image_height = original_image.size
        color_mode = original_image.mode

        image = original_image.convert("RGB")

    except Exception:
        st.error("Unable to read this image.")
        st.stop()

    # Reset results when a new image is uploaded
    if st.session_state.get("current_fingerprint") != fingerprint:

        st.session_state["current_fingerprint"] = fingerprint

        for key in [
            "detector_one_result",
            "detector_one_error",
            "detector_two_result",
            "detector_two_error"
        ]:
            st.session_state.pop(key, None)

    # ======================================
    # FILE INFORMATION
    # ======================================

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)
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
    # METADATA
    # ======================================

    st.divider()
    st.subheader("📋 Metadata Inspection")

    try:
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
                "that the image is AI-generated."
            )

    except Exception:
        exif_data = {}
        st.warning("Metadata could not be read.")

    # ======================================
    # AI DETECTOR SECTION
    # ======================================

    st.divider()
    st.subheader("🤖 AI Image Detection")

    st.write(
        "VERISCOPE uses two different research models "
        "to compare image classifications."
    )

    st.markdown("### Detector 1 — Dima806")

    if st.button(
        "Analyse with Detector 1",
        use_container_width=True
    ):

        try:
            with st.spinner("Running Detector 1..."):

                detector = load_detector_one()
                results = detector(image)

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
                        f"Unexpected labels: {results}"
                    )

                st.session_state["detector_one_result"] = {
                    "ai": float(ai_result["score"]),
                    "human": float(human_result["score"])
                }

                st.session_state.pop("detector_one_error", None)

        except Exception as error:
            st.session_state["detector_one_error"] = str(error)

    if "detector_one_result" in st.session_state:

        result = st.session_state["detector_one_result"]

        a, b = st.columns(2)

        a.metric("AI-generated", f"{result['ai'] * 100:.2f}%")
        b.metric("Human-created", f"{result['human'] * 100:.2f}%")

    elif "detector_one_error" in st.session_state:

        st.error(
            "Detector 1 failed: "
            + st.session_state["detector_one_error"][:400]
        )

    # ======================================
    # SECOND DETECTOR
    # ======================================

    st.markdown("### Detector 2 — OpenFake SwinV2")

    st.caption(
        "This detector uses a SwinV2 image-classification model. "
        "Its model card specifies a temperature scaling factor of 2."
    )

    if st.button(
        "Analyse with Detector 2",
        use_container_width=True
    ):

        try:
            with st.spinner(
                "Loading OpenFake model and analysing image. "
                "The first download may take several minutes..."
            ):

                processor, model = load_detector_two()

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

                ai_score = float(probabilities[1].item())
                human_score = float(probabilities[0].item())

                st.session_state["detector_two_result"] = {
                    "ai": ai_score,
                    "human": human_score
                }

                st.session_state.pop("detector_two_error", None)

        except Exception as error:

            st.session_state["detector_two_error"] = str(error)

    if "detector_two_result" in st.session_state:

        result = st.session_state["detector_two_result"]

        a, b = st.columns(2)

        a.metric("AI-generated", f"{result['ai'] * 100:.2f}%")
        b.metric("Human-created", f"{result['human'] * 100:.2f}%")

    elif "detector_two_error" in st.session_state:

        st.error(
            "Detector 2 failed: "
            + st.session_state["detector_two_error"][:400]
        )

    # ======================================
    # COMPARISON
    # ======================================

    st.divider()
    st.subheader("📊 Detector Comparison")

    one = st.session_state.get("detector_one_result")
    two = st.session_state.get("detector_two_result")

    if one and two:

        one_label = (
            "AI-generated" if one["ai"] >= 0.5 else "Human-created"
        )

        two_label = (
            "AI-generated" if two["ai"] >= 0.5 else "Human-created"
        )

        comparison_col1, comparison_col2 = st.columns(2)

        with comparison_col1:
            st.write("**Detector 1**")
            st.write(one_label)
            st.progress(one["ai"])

        with comparison_col2:
            st.write("**Detector 2**")
            st.write(two_label)
            st.progress(two["ai"])

        if one_label == two_label:

            st.success(
                f"Both models agree: {one_label}."
            )

        else:

            st.warning(
                "MODEL DISAGREEMENT: The detectors produced "
                "different classifications. Manual verification "
                "is recommended."
            )

        st.caption(
            "Agreement does not prove correctness. The models "
            "may share weaknesses and can both be wrong."
        )

    elif one or two:

        st.info(
            "Run both detectors to see their comparison."
        )

    else:

        st.info(
            "No detector results yet. Run both analyses above."
        )

    # ======================================
    # VERIFICATION REPORT
    # ======================================

    st.divider()
    st.subheader("📄 Verification Report")

    def result_text(result):
        if result is None:
            return "Not completed"

        return (
            f"AI-generated: {result['ai'] * 100:.2f}%\n"
            f"Human-created: {result['human'] * 100:.2f}%"
        )

    report = f"""
VERISCOPE - DIGITAL CONTENT VERIFICATION REPORT

Date: {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}

FILE INFORMATION
Name: {uploaded.name}
Format: {image_format}
Dimensions: {image_width} x {image_height}
Size: {len(image_bytes)} bytes

SHA-256
{fingerprint}

METADATA
EXIF present: {"Yes" if exif_data else "No"}

DETECTOR 1 - DIMA806
{result_text(one)}

DETECTOR 2 - OPENFAKE SWINV2
{result_text(two)}

DISCLAIMER
These are automated model estimates, not proof of authenticity.

Model agreement does not guarantee correctness.
Images should be evaluated with additional evidence.

OpenFake model licence: CC-BY-NC-4.0.
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
        <p>Upload an image to begin digital verification.</p>
        <p class="small">
        Two AI Detectors • Metadata Inspection • SHA-256
        Fingerprinting • Detector Comparison • Reports
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
