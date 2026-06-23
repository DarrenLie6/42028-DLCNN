import streamlit as st
import numpy as np
from PIL import Image
import io
import sys
import os
import uuid

from pathlib import Path
import rasterio
import tempfile

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app.inference import DamageAssessor, find_best_checkpoint
from app.llm_feedback import compute_damage_stats, stream_damage_feedback

st.set_page_config(
    page_title="ImpactVision",
    layout="wide",
    page_icon="🛰️"
)

BACKGROUND_URL = (
    "https://raw.githubusercontent.com/DarrenLie6/42028-DLCNN/"
    "master/app/assets/4.jpg"
)


def _find_background() -> Path | None:
    assets = ROOT_DIR / "app" / "assets"
    for name in ("background.jpg", "background.jpeg", "background.png"):
        p = assets / name
        if p.exists():
            return p
    return None

EXAMPLES_DIR = ROOT_DIR / "app" / "assets" / "examples"

EXAMPLE_SCENES = [
    {"name": "Hurricane Matthew",      "post": "hurricane-matthew_00000027_post_disaster.png", "pre": "hurricane-matthew_00000027_pre_disaster.png"},
    {"name": "Joplin Tornado 1",    "post": "joplin-tornado_00000089_post_disaster.png", "pre": "joplin-tornado_00000089_pre_disaster.png"}, #tif
    {"name": "Joplin Tornado 2", "post": "joplin-tornado_00000127_post_disaster.png", "pre": "joplin-tornado_00000127_pre_disaster.png"}, #tif
    {"name": "Midwest Flooding",      "post": "midwest-flooding_00000172_post_disaster.png", "pre": "midwest-flooding_00000172_pre_disaster.png"}, #tif
    {"name": "Palu Tsunami 1",    "post": "palu-tsunami_00000112_post_disaster.png", "pre": "palu-tsunami_00000112_pre_disaster.png"}, #tif
    {"name": "Palu Tsunami 2",    "post": "palu-tsunami_00000165_post_disaster.png", "pre": "palu-tsunami_00000165_pre_disaster.png"},
    {"name": "Santa Rosa Wildfire", "post": "santa-rosa-wildfire_00000038_post_disaster.png", "pre": "santa-rosa-wildfire_00000038_pre_disaster.png"},
    {"name": "Socal Fire 1",      "post": "socal-fire_00001252_post_disaster.png", "pre": "socal-fire_00001252_pre_disaster.png"},
    {"name": "Socal Fire 2",    "post": "socal-fire_00001236_post_disaster.png", "pre": "socal-fire_00001236_pre_disaster.png"},

]

def load_example_image(filename):
    path = EXAMPLES_DIR / filename
    return Image.open(path).convert("RGB")


def apply_theme(card_opacity: float = 0.92) -> None:
    # bg = _find_background()
    # if bg is not None:
    #     ext = bg.suffix.lstrip(".").lower()
    #     mime = "jpeg" if ext == "jpg" else ext
    #     b64 = base64.b64encode(bg.read_bytes()).decode()
    #     image_src = f"data:image/{mime};base64,{b64}"
    # elif BACKGROUND_URL:
    #     image_src = BACKGROUND_URL
    # else:
    #     image_src = None

    # if image_src is not None:
    #     background_rule = (
    #         f'background-image: url("{image_src}");'
    #         "background-size: cover;"
    #         "background-position: center;"
    #         "background-attachment: fixed;"
    #     )
    # else:
    background_rule = "background: linear-gradient(135deg, #1e3a5f 0%, #2c5364 100%);"

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            {background_rule}
        }}
        [data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0);
        }}
        [data-testid="stAppViewContainer"] .block-container {{
            background: rgba(255, 255, 255, {card_opacity});
            padding: 2rem 2.5rem 2.5rem 2.5rem;
            border-radius: 18px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.35);
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
            max-width: 1400px;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0;
            padding: 8px 20px;
            font-weight: 600;
        }}
        .result-placeholder {{
            background: rgba(0,0,0,0.04);
            border: 2px dashed rgba(0,0,0,0.15);
            border-radius: 12px;
            height: 300px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: rgba(0,0,0,0.35);
            font-size: 0.95rem;
            gap: 0.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_theme()

BITEMPORAL_DIR = ROOT_DIR / "checkpoints" / "semantic_seg_transformer"
POSTONLY_DIR   = ROOT_DIR / "checkpoints" / "semantic_seg_transformer_post"


@st.cache_resource
def load_assessor():
    bi   = find_best_checkpoint(BITEMPORAL_DIR)
    post = find_best_checkpoint(POSTONLY_DIR)
    if bi is None or post is None:
        st.error(
            f"Missing checkpoints.\n"
            f"  bi-temporal: {bi}\n  post-only: {post}\n"
            f"Train the models first or check the checkpoint directories."
        )
        st.stop()
    return DamageAssessor(bitemporal_ckpt=bi, postonly_ckpt=post)


assessor = load_assessor()

COLOR_MAP = {
    0: [0, 0, 0, 0],
    1: [46, 204, 113, 150],
    2: [241, 196, 15, 150],
    3: [231, 76, 60, 150],
}


def load_image_from_bytes(file_bytes, file_name):
    file_name_lower = file_name.lower()
    if file_name_lower.endswith((".tif", ".tiff")):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            with rasterio.open(tmp_path) as src:
                data = src.read([1, 2, 3] if src.count >= 3 else list(range(1, src.count + 1)))
                data = np.transpose(data, (1, 2, 0))
                if data.dtype == np.uint8:
                    img_array = data
                elif data.dtype == np.uint16:
                    img_array = (data / 256).astype(np.uint8)
                else:
                    data_min, data_max = data.min(), data.max()
                    if data_max > data_min:
                        img_array = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
                    else:
                        img_array = (data * 255).astype(np.uint8)
                if img_array.shape[2] == 1:
                    return Image.fromarray(img_array[:, :, 0]).convert("RGB")
                return Image.fromarray(img_array).convert("RGB")
        finally:
            os.unlink(tmp_path)
    else:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def overlay_mask(post_img: Image.Image, mask: np.ndarray) -> Image.Image:
    w, h = post_img.size
    rgba_mask = np.zeros((h, w, 4), dtype=np.uint8)
    for class_idx, color in COLOR_MAP.items():
        rgba_mask[mask == class_idx] = color
    mask_img = Image.fromarray(rgba_mask)
    bg = post_img.convert("RGBA")
    if bg.size != (w, h):
        bg = bg.resize((w, h))
    return Image.alpha_composite(bg, mask_img).convert("RGB")


def render_statistics(mask, mode):
    building_pixels = int((mask > 0).sum())
    if building_pixels == 0:
        building_pixels = 1
    intact_pct    = (int((mask == 1).sum()) / building_pixels) * 100
    damaged_pct   = (int((mask == 2).sum()) / building_pixels) * 100
    destroyed_pct = (int((mask == 3).sum()) / building_pixels) * 100

    c1, c2, c3 = st.columns(3)
    for col, emoji, pct, label, bg, border in [
        (c1, "🟩", intact_pct,    "Intact",    "rgba(46,204,113,0.15)", "rgba(46,204,113,0.5)"),
        (c2, "🟨", damaged_pct,   "Damaged",   "rgba(241,196,15,0.15)", "rgba(241,196,15,0.5)"),
        (c3, "🟥", destroyed_pct, "Destroyed", "rgba(231,76,60,0.15)",  "rgba(231,76,60,0.5)"),
    ]:
        col.markdown(
            f"""
            <div style="
                background:{bg};
                border:2px solid {border};
                border-radius:12px;
                padding:0.8rem;
                text-align:center;
            ">
                <p style="margin:0;font-size:1.5rem;">{emoji}</p>
                <p style="margin:0;font-weight:700;font-size:1.2rem;">{pct:.1f}%</p>
                <p style="margin:0;font-size:0.85rem;color:#555;">{label}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_placeholder():
    st.markdown(
        """
        <div class="result-placeholder">
            <span style="font-size:2rem;">🗺️</span>
            <span>Damage map will appear here</span>
            <span style="font-size:0.8rem;">Upload an image and click Generate</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(result):
    st.image(
        result["result_map"],
        caption=f"Damage Map — {result['mode']} model",
        use_container_width=True
    )
    st.markdown(
        "<div style='font-size:0.85rem;margin-bottom:0.8rem;'>"
        "🟩 Intact &nbsp;&nbsp; 🟨 Damaged &nbsp;&nbsp; 🟥 Destroyed"
        "</div>",
        unsafe_allow_html=True,
    )
    render_statistics(result["mask"], result["mode"])

    # AI interpretation layer: turn the model's per-class stats into a
    # disaster-response narrative via the local LLM. Gated behind a button and
    # cached per-result so it isn't re-run on every Streamlit rerun.
    st.markdown("---")
    ai_key = f"ai_{result['id']}"
    if st.button(
        "Generate AI Assessment",
        key=f"ai_btn_{result['id']}",
        use_container_width=True,
    ):
        stats = compute_damage_stats(result["mask"])
        image = result["result_map"]
        # Release the CNN's reserved-but-unused VRAM so Ollama can load the LLM
        # onto the GPU with maximum headroom.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            st.session_state[ai_key] = st.write_stream(
                stream_damage_feedback(stats, result["mode"], image=image)
            )
        except Exception as e:  # missing model / Ollama not running / etc.
            st.error(str(e))
    elif ai_key in st.session_state:
        st.markdown(st.session_state[ai_key])

    buf = io.BytesIO()
    result["result_map"].save(buf, format="PNG")
    buf.seek(0)
    st.download_button(
        label="Download Damage Map",
        data=buf.getvalue(),
        file_name=f"damage_map_{result['mode']}_{result['name']}",
        mime="image/png",
        key=f"dl_{result['mode']}_{result['name']}"
    )


# header
st.title("🛰️ ImpactVision")
st.caption("Upload post-disaster satellite imagery to generate a colour-graded building damage map.")

tab_post, tab_bi = st.tabs(["Post-Only", "Pre + Post (Change Detection)"])

# post only tab
with tab_post:
    with st.expander("Try an example scene", expanded=False):
        st.caption("Click a thumbnail to instantly load a sample disaster scene.")
        cols = st.columns(len(EXAMPLE_SCENES))
        selected_example = None

        for i, scene in enumerate(EXAMPLE_SCENES):
            with cols[i]:
                thumb = load_example_image(scene["post"])
                thumb.thumbnail((150, 150))  # small thumbnail, fast to render
                st.image(thumb, use_container_width=True)
                if st.button(scene["name"], key=f"example_{i}", use_container_width=True):
                    selected_example = scene

        if selected_example is not None:
            st.session_state["post_example"] = selected_example
            st.session_state.pop("post_result", None)
    
    # st.caption("Quick demo: select a sample scene, or upload your own image below.")
    
    # example_names = ["None — upload my own"] + [s["name"] for s in EXAMPLE_SCENES]
    # choice = st.selectbox("Try an example", example_names, key="example_select")
    
    # if choice != "None — upload my own":
    #     selected_example = next(s for s in EXAMPLE_SCENES if s["name"] == choice)
    #     st.session_state["post_example"] = selected_example
    # else:
    #     st.session_state.pop("post_example", None)

    st.write("---")
    st.subheader("Upload Imagery")
    st.caption("Or upload your own post-disaster satellite image.")
    
    post_file = st.file_uploader(
        "Drag & Drop Post-Event Image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="post_only"
    )
    
    # if example clicked, override with example image
    if selected_example is not None:
        st.session_state["post_example"] = selected_example
        st.session_state.pop("post_result", None)  # clear old result
    
    use_example = "post_example" in st.session_state and post_file is None

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.subheader("Image Preview")

        if post_file is not None:
            post_img_preview = load_image_from_bytes(post_file.getvalue(), post_file.name)
            preview_name = post_file.name
        elif use_example:
            post_img_preview = load_example_image(st.session_state["post_example"]["post"])
            preview_name = st.session_state["post_example"]["name"]
        else:
            post_img_preview = None
            preview_name = None

        if post_img_preview is not None:
            st.image(post_img_preview, caption=preview_name, use_container_width=True)

        run_post = st.button(
            "🚀 GENERATE COLOUR-GRADED MAP",
            use_container_width=True,
            key="run_post",
            disabled=post_img_preview is None
        )

    with right_col:
        st.subheader("Assessment Results")

        if run_post and post_img_preview is not None:
            progress = st.progress(0, text="Loading image...")
            post_img = post_img_preview
            progress.progress(33, text="Running model...")
            mask, mode = assessor.predict(post_img, pre_img=None)
            progress.progress(66, text="Generating damage map...")
            result_map = overlay_mask(post_img, mask)
            progress.progress(100, text="Complete!")
            progress.empty()
            st.session_state["post_result"] = {
                "id":         uuid.uuid4().hex,
                "name":       preview_name,
                "post_img":   post_img,
                "result_map": result_map,
                "mask":       mask,
                "mode":       mode
            }

        if "post_result" in st.session_state:
            render_result(st.session_state["post_result"])
        else:
            render_placeholder()

# bitemporal tab
with tab_bi:
    with st.expander("Try an example scene", expanded=False):
        st.caption("Click a thumbnail to instantly load a sample pre/post pair.")
        cols = st.columns(len(EXAMPLE_SCENES))
        selected_example_bi = None

        for i, scene in enumerate(EXAMPLE_SCENES):
            with cols[i]:
                thumb = load_example_image(scene["post"])
                thumb.thumbnail((150, 150))
                st.image(thumb, use_container_width=True)
                if st.button(scene["name"], key=f"example_bi_{i}", use_container_width=True):
                    selected_example_bi = scene

        if selected_example_bi is not None:
            st.session_state["bi_example"] = selected_example_bi
            st.session_state.pop("bi_result", None)

    st.write("---")
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.subheader("Upload Imagery")
        st.caption("Or upload your own matching pre and post-disaster image pair.")

        col_pre, col_post = st.columns(2)
        with col_pre:
            st.write("**Pre-Event**")
            pre_file = st.file_uploader(
                "Drag & Drop Pre-Event",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                key="pre_bi"
            )

        with col_post:
            st.write("**Post-Event**")
            post_file_bi = st.file_uploader(
                "Drag & Drop Post-Event",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                key="post_bi"
            )

        # use uploaded files if present, otherwise fall back to selected example
        use_example_bi = (
            "bi_example" in st.session_state
            and pre_file is None
            and post_file_bi is None
        )

        if pre_file is not None and post_file_bi is not None:
            pre_img_preview  = load_image_from_bytes(pre_file.getvalue(), pre_file.name)
            post_img_preview = load_image_from_bytes(post_file_bi.getvalue(), post_file_bi.name)
            preview_name_bi  = post_file_bi.name
        elif use_example_bi:
            example = st.session_state["bi_example"]
            pre_img_preview  = load_example_image(example["pre"])
            post_img_preview = load_example_image(example["post"])
            preview_name_bi  = example["name"]
        else:
            pre_img_preview  = None
            post_img_preview = None
            preview_name_bi  = None

        if pre_img_preview is not None:
            with col_pre:
                st.image(pre_img_preview, caption="Pre-Event", use_container_width=True)
        if post_img_preview is not None:
            with col_post:
                st.image(post_img_preview, caption="Post-Event", use_container_width=True)

        bi_ready = pre_img_preview is not None and post_img_preview is not None

        run_bi = st.button(
            "🚀 GENERATE COLOUR-GRADED MAP",
            use_container_width=True,
            key="run_bi",
            disabled=not bi_ready
        )

    with right_col:
        st.subheader("Assessment Results")

        if run_bi and bi_ready:
            progress = st.progress(0, text="Loading images...")
            post_img = post_img_preview
            pre_img  = pre_img_preview
            progress.progress(33, text="Running model...")
            mask, mode = assessor.predict(post_img, pre_img)
            progress.progress(66, text="Generating damage map...")
            result_map = overlay_mask(post_img, mask)
            progress.progress(100, text="Complete!")
            progress.empty()
            st.session_state["bi_result"] = {
                "id":         uuid.uuid4().hex,
                "name":       preview_name_bi,
                "post_img":   post_img,
                "result_map": result_map,
                "mask":       mask,
                "mode":       mode
            }

        if "bi_result" in st.session_state:
            render_result(st.session_state["bi_result"])
        else:
            render_placeholder()