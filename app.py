import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pickle
from PIL import Image
import base64
import io
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from urllib.parse import quote
from datetime import datetime
import tempfile
import os
import random
import re
import json

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="GlucoTrack — Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ==============================
# SESSION STATE
# ==============================
default_values = {
    "logged_in": False,
    "theme": "Light",
    "patient_photo": None,
    "current_user": None,
    "selected_menu": "🏠 Welcome",
    "otp_sent": False,
    "login_otp": None,
    "login_identifier": "",
    "login_username": None,
    "active_patient_name": "",
    "active_patient_id": "",
    "active_patient_age": None,
    "active_patient_gender": "",
    "patients_db": {},
    "previous_menu": None,
    "page_history": [],
}

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ==============================
# SIMPLE LOCAL STORAGE
# ==============================
# This keeps users and enrolled patients after page refresh.
# For a real app, replace this JSON storage with a proper database.
DATA_FILE = "glucotrack_data.json"

def load_local_data():
    default_data = {
        "users_db": {
            "admin": {
                "password": "1234",
                "full_name": "Administrator",
                "role": "Doctor",
                "account_type": "Doctor",
                "email": "admin@gmail.com",
                "phone": "9999999999",
                "specialization": "General Physician",
                "license": "DEMO-ADMIN"
            }
        },
        "patients_db": {}
    }
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            default_data["users_db"].update(saved.get("users_db", {}))
            default_data["patients_db"].update(saved.get("patients_db", {}))
    except Exception:
        pass
    return default_data

def save_local_data():
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "users_db": st.session_state.users_db,
                "patients_db": st.session_state.patients_db
            }, f, indent=2)
    except Exception:
        pass

local_data = load_local_data()
st.session_state.users_db = local_data["users_db"]
st.session_state.patients_db = local_data["patients_db"]

# Page names used for browser back/forward support.
PAGE_SLUGS = {
    "🏠 Welcome": "welcome",
    "🔐 Login": "login",
    "📝 Sign Up": "signup",
    "👋 Doctor Home": "doctor_home",
    "📋 Enroll Patient": "enroll_patient",
    "👥 Patient Details": "patient_details",
    "🔬 Prediction": "prediction",
    "📊 Visualization": "visualization",
    "ℹ️ About": "about",
}
SLUG_PAGES = {v: k for k, v in PAGE_SLUGS.items()}

try:
    page_from_url = st.query_params.get("page", None)
    if page_from_url in SLUG_PAGES:
        st.session_state.selected_menu = SLUG_PAGES[page_from_url]
except Exception:
    pass

# Restore login on browser refresh for this demo app.
try:
    remembered_user = st.query_params.get("user", None)
    if remembered_user and remembered_user in st.session_state.users_db and not st.session_state.logged_in:
        st.session_state.logged_in = True
        st.session_state.current_user = remembered_user
        if st.session_state.selected_menu == "🏠 Welcome":
            role = st.session_state.users_db[remembered_user].get("role", "Patient")
            st.session_state.selected_menu = "👋 Doctor Home" if role == "Doctor" else "🔬 Prediction"
except Exception:
    pass

# ==============================
# THEME CSS
# ==============================
def apply_theme(theme):
    is_dark = theme == "Dark"

    bg = "#07111F" if is_dark else "#F7FCFF"
    sidebar_bg = "#0D1B2F" if is_dark else "#E7F6FF"
    card_bg = "#101B33" if is_dark else "#FFFFFF"
    text_main = "#F8FAFC" if is_dark else "#0B1F33"
    text_sub = "#B8C7E6" if is_dark else "#486174"
    text_input = "#F8FAFC" if is_dark else "#0B1F33"
    accent = "#38BDF8" if is_dark else "#0077B6"
    accent2 = "#2DD4BF" if is_dark else "#00B4D8"
    border = "#263B5D" if is_dark else "#BFE8F8"
    input_bg = "#13213C" if is_dark else "#FFFFFF"
    hover_bg = "#1D3155" if is_dark else "#DDF4FF"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600;700&display=swap');

    html, body, .stApp {{
        background: radial-gradient(circle at top left, rgba(56,189,248,0.18), transparent 34%), linear-gradient(135deg, {bg}, {hover_bg}) !important;
        color: {text_main} !important;
        font-family: 'DM Sans', sans-serif;
    }}

    .block-container {{
        max-width: 1180px !important;
        padding-top: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {sidebar_bg}, {hover_bg}) !important;
        border-right: 1px solid {border};
    }}

    section[data-testid="stSidebar"] * {{
        color: {text_main} !important;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {text_main} !important;
        font-family: 'DM Serif Display', serif;
    }}

    p, span, div, label {{
        color: {text_main} !important;
    }}

    .stMarkdown p {{
        color: {text_sub} !important;
        font-size: 0.95rem;
        line-height: 1.7;
    }}

    input, textarea {{
        background-color: {input_bg} !important;
        color: {text_input} !important;
        border: 1px solid {border} !important;
        border-radius: 10px !important;
    }}

    input::placeholder, textarea::placeholder {{
        color: {text_sub} !important;
        opacity: 0.7 !important;
    }}

    .stTextInput label,
    .stNumberInput label,
    .stTextArea label,
    .stSelectbox label,
    .stRadio label,
    .stFileUploader label {{
        color: {text_sub} !important;
        font-size: 0.84rem;
        font-weight: 600;
    }}

    div[data-baseweb="select"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] div {{
        background-color: {input_bg} !important;
        color: {text_input} !important;
        border-color: {border} !important;
    }}

    div[data-baseweb="select"] svg {{
        fill: {text_input} !important;
    }}

    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    ul[role="listbox"],
    div[role="listbox"] {{
        background-color: {input_bg} !important;
        color: {text_input} !important;
        border: 1px solid {border} !important;
        border-radius: 10px !important;
    }}

    li[role="option"],
    div[role="option"] {{
        background-color: {input_bg} !important;
        color: {text_input} !important;
    }}

    li[role="option"] *,
    div[role="option"] * {{
        color: {text_input} !important;
    }}

    li[role="option"]:hover,
    div[role="option"]:hover,
    li[role="option"][aria-selected="true"],
    div[role="option"][aria-selected="true"] {{
        background-color: {hover_bg} !important;
        color: {text_input} !important;
    }}

    .stFileUploader,
    .stFileUploader section,
    .stFileUploader section > div,
    div[data-testid="stFileUploader"],
    div[data-testid="stFileUploader"] section,
    div[data-testid="stFileUploader"] section div {{
        background-color: {input_bg} !important;
        color: {text_input} !important;
        border-color: {border} !important;
    }}

    .stFileUploader button,
    div[data-testid="stFileUploader"] button {{
        background: linear-gradient(135deg, {accent}, {accent2}) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        cursor: pointer !important;
    }}

    .stFileUploader small,
    .stFileUploader span,
    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] span {{
        color: {text_sub} !important;
    }}

    .stRadio div[role="radiogroup"] label {{
        background-color: {input_bg} !important;
        color: {text_input} !important;
        border-radius: 10px !important;
        padding: 8px 12px !important;
    }}

    .stRadio div[role="radiogroup"] label:hover {{
        background-color: {hover_bg} !important;
    }}

    .stButton > button {{
        cursor: pointer !important;
        background: linear-gradient(135deg, {accent}, {accent2}) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 999px !important;
        padding: 0.38rem 0.95rem !important;
        min-height: 34px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.86rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
        transition: all 0.2s ease !important;
        box-shadow: 0 8px 22px rgba(2,132,199,0.24) !important;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px) scale(1.01) !important;
        box-shadow: 0 12px 28px rgba(2,132,199,0.35) !important;
        filter: brightness(1.06) !important;
    }}

    a, button, [role="button"], .stRadio label, .stSelectbox [role="option"] {{
        cursor: pointer !important;
    }}

    .card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(17,24,39,0.08);
    }}

    .metric-tile {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }}

    .metric-tile .val {{
        font-size: 2rem;
        font-weight: 800;
        color: {accent} !important;
    }}

    .metric-tile .lbl {{
        font-size: 0.78rem;
        color: {text_sub} !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }}

    .hero-title {{
        font-family: 'DM Serif Display', serif;
        font-size: 3rem;
        line-height: 1.15;
        color: {text_main} !important;
    }}

    .hero-sub {{
        font-size: 1.05rem;
        color: {text_sub} !important;
        max-width: 650px;
        line-height: 1.75;
    }}

    .accent-line {{
        display: inline-block;
        height: 4px;
        width: 66px;
        background: linear-gradient(135deg, {accent}, {accent2});
        border-radius: 4px;
        margin-bottom: 1rem;
    }}

    .badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }}

    .badge-green {{ background: rgba(34,197,94,0.14); color: #22C55E !important; }}
    .badge-red {{ background: rgba(239,68,68,0.14); color: #EF4444 !important; }}

    hr {{
        border-color: {border} !important;
    }}

    .stSuccess, .stError, .stWarning, .stInfo {{
        border-radius: 12px !important;
    }}

    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        border-radius: 12px !important;
        padding: 0.65rem 0.85rem !important;
        margin: 0.25rem 0 !important;
        border: 1px solid transparent !important;
        transition: all 0.2s ease !important;
    }}

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
        background: rgba(2,132,199,0.16) !important;
        border-color: {border} !important;
    }}

    .feature-card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 18px;
        padding: 1.2rem;
        min-height: 155px;
        box-shadow: 0 10px 30px rgba(15,23,42,0.08);
    }}

    .feature-card .feature-value {{
        font-size: 1.6rem;
        font-weight: 800;
        color: {accent} !important;
        margin-bottom: 0.35rem;
    }}

    .feature-card .feature-title {{
        font-size: 0.85rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: {text_main} !important;
        margin-bottom: 0.4rem;
    }}

    .feature-card .feature-desc {{
        font-size: 0.85rem;
        line-height: 1.55;
        color: {text_sub} !important;
    }}


    .welcome-hero {{
        background: linear-gradient(135deg, rgba(29,78,216,0.14), rgba(20,184,166,0.12));
        border: 1px solid {border};
        border-radius: 28px;
        padding: 3rem 3.2rem;
        box-shadow: 0 24px 70px rgba(15,23,42,0.12);
        position: relative;
        overflow: hidden;
    }}

    .welcome-hero:after {{
        content: "";
        position: absolute;
        right: -70px;
        top: -70px;
        width: 240px;
        height: 240px;
        border-radius: 50%;
        background: linear-gradient(135deg, {accent}, {accent2});
        opacity: 0.14;
    }}

    .brand-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.5rem 0.9rem;
        border-radius: 999px;
        background: rgba(20,184,166,0.13);
        color: {text_main} !important;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }}

    .hero-actions {{
        margin-top: 1.5rem;
        font-size: 0.92rem;
        color: {text_sub} !important;
    }}

    section[data-testid="stSidebar"] {{
        box-shadow: 12px 0 35px rgba(15,23,42,0.08);
    }}

    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb] {{
        background: transparent !important;
    }}


    .welcome-hero-center {{
        max-width: 980px;
        margin: 0 auto;
        text-align: center;
        padding: 2.5rem 1rem 1rem 1rem;
    }}
    .welcome-text-area {{
        max-width: 760px;
        margin: 0 auto 1.5rem auto;
    }}
    .welcome-image-card {{
        max-width: 760px;
        height: 330px;
        margin: 1.6rem auto 0 auto;
        border-radius: 26px;
        overflow: hidden;
        border: 1px solid {border};
        box-shadow: 0 24px 70px rgba(15,23,42,0.16);
        background: linear-gradient(135deg, {accent}, {accent2});
    }}
    .welcome-image-card img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
        opacity: 0.94;
    }}



    /* Keep dataframe and chart toolbar icons visible in dark mode */
    div[data-testid="stElementToolbar"] {{
        background: rgba(255,255,255,0.92) !important;
        border-radius: 10px !important;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12) !important;
    }}
    div[data-testid="stElementToolbar"] button,
    div[data-testid="stElementToolbar"] svg {{
        color: #0F172A !important;
        fill: #0F172A !important;
        stroke: #0F172A !important;
        opacity: 1 !important;
    }}
    div[data-testid="stDataFrame"] button svg,
    div[data-testid="stDataFrame"] svg {{
        color: #0F172A !important;
        fill: #0F172A !important;
        stroke: #0F172A !important;
    }}



    /* Professional compact controls */
    .stButton > button, .stDownloadButton > button {{
        min-height: 34px !important;
        padding: 0.36rem 0.9rem !important;
        border-radius: 999px !important;
        font-size: 0.84rem !important;
    }}

    div[data-testid="stNumberInput"] button {{
        background: {hover_bg} !important;
        color: {text_main} !important;
        border: 1px solid {border} !important;
        border-radius: 8px !important;
    }}
    div[data-testid="stNumberInput"] button svg,
    div[data-testid="stNumberInput"] svg {{
        color: {text_main} !important;
        fill: {text_main} !important;
        stroke: {text_main} !important;
        opacity: 1 !important;
    }}

    button[data-testid="stBaseButton-headerNoPadding"],
    button[kind="headerNoPadding"],
    div[data-testid="stSidebarCollapsedControl"] button,
    div[data-testid="stSidebarCollapsedControl"] svg {{
        color: {text_main} !important;
        fill: {text_main} !important;
        stroke: {text_main} !important;
        background: {card_bg} !important;
        border-radius: 10px !important;
        opacity: 1 !important;
    }}

    .responsive-card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 18px;
        padding: 1.2rem;
        box-shadow: 0 12px 32px rgba(15,23,42,0.08);
    }}

    .back-shell {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 0.25rem 0 1.25rem 0;
        padding: 0.85rem 1rem;
        background: linear-gradient(135deg, rgba(14,165,233,0.12), rgba(45,212,191,0.10));
        border: 1px solid {border};
        border-radius: 18px;
        box-shadow: 0 12px 30px rgba(15,23,42,0.08);
    }}

    .back-icon {{
        width: 38px;
        height: 38px;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, {accent}, {accent2});
        color: #FFFFFF !important;
        font-size: 1rem;
        font-weight: 900;
        box-shadow: 0 10px 22px rgba(2,132,199,0.25);
    }}

    .back-title {{
        font-size: 0.95rem;
        font-weight: 800;
        color: {text_main} !important;
        margin-bottom: 0.08rem;
    }}

    .back-subtitle {{
        font-size: 0.78rem;
        color: {text_sub} !important;
        line-height: 1.35;
    }}

    div[data-testid="column"]:has(.back-button-holder) .stButton > button {{
        background: {card_bg} !important;
        color: {accent} !important;
        border: 1px solid {border} !important;
        box-shadow: 0 10px 22px rgba(15,23,42,0.08) !important;
        font-weight: 800 !important;
    }}

    @media (max-width: 900px) {{
        .block-container {{ padding-left: 1rem !important; padding-right: 1rem !important; }}
        .hero-title {{ font-size: 2.15rem !important; }}
        .welcome-image-card {{ height: 230px !important; }}
        .card, .feature-card, .responsive-card {{ padding: 1rem !important; }}
        div[data-testid="column"] {{ width: 100% !important; flex: 1 1 100% !important; }}
        .stDataFrame {{ overflow-x: auto !important; }}
    }}

    @media (min-width: 901px) and (max-width: 1200px) {{
        .block-container {{ padding-left: 2rem !important; padding-right: 2rem !important; }}
        .hero-title {{ font-size: 2.55rem !important; }}
    }}

    </style>
    """, unsafe_allow_html=True)

# ==============================
# LOAD MODEL & COLUMNS
# ==============================
@st.cache_resource
def load_model():
    model = pickle.load(open("diabetes_model.pkl", "rb"))
    columns = pickle.load(open("columns.pkl", "rb"))
    return model, columns

model, columns = load_model()

# ==============================
# HELPER FUNCTIONS
# ==============================
def go_to_page(page_name, add_to_history=True):
    current_page = st.session_state.get("selected_menu")
    if add_to_history and current_page and current_page != page_name:
        history = st.session_state.get("page_history", [])
        if not history or history[-1] != current_page:
            history.append(current_page)
        st.session_state.page_history = history[-12:]

    st.session_state.selected_menu = page_name
    try:
        st.query_params["page"] = PAGE_SLUGS.get(page_name, "")
        if st.session_state.get("current_user"):
            st.query_params["user"] = st.session_state.current_user
    except Exception:
        pass
    st.rerun()

def get_safe_back_page():
    history = st.session_state.get("page_history", [])
    while history:
        previous_page = history.pop()
        if previous_page != st.session_state.get("selected_menu"):
            st.session_state.page_history = history
            return previous_page

    if st.session_state.get("logged_in", False):
        current = get_current_user_data() if "users_db" in st.session_state else {}
        if current.get("role") == "Doctor":
            return "👋 Doctor Home"
        return "🔬 Prediction"
    return "🏠 Welcome"

def go_back_page():
    back_page = get_safe_back_page()
    if back_page == st.session_state.get("selected_menu"):
        back_page = "🏠 Welcome"
    go_to_page(back_page, add_to_history=False)

def show_professional_back_button():
    if st.session_state.get("selected_menu") == "🏠 Welcome":
        return

    left_col, right_col = st.columns([4, 1])
    with left_col:
        st.markdown('''
            <div class="back-shell">
                <div class="back-icon">←</div>
                <div>
                    <div class="back-title">Back Navigation</div>
                    <div class="back-subtitle">Return to the previous screen without losing the dashboard style.</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
    with right_col:
        st.markdown('<div class="back-button-holder"></div>', unsafe_allow_html=True)
        back_key = "back_btn_" + PAGE_SLUGS.get(st.session_state.get("selected_menu"), "page")
        if st.button("← Back", use_container_width=True, key=back_key):
            go_back_page()

def generate_otp():
    return str(random.randint(100000, 999999))


def clean_phone(value):
    return re.sub(r"\D", "", value or "")




def build_whatsapp_file_share_button(pdf_bytes, file_name, caption):
    """Create a browser share button for the generated PDF.
    Works on mobile browsers that support Web Share API file sharing.
    Desktop WhatsApp Web may still require manual attachment.
    """
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    safe_caption = caption.replace("`", "'").replace("\\", "\\\\")
    components.html(f"""
    <div style="margin-top:12px;">
      <button id="sharePdfBtn" style="
        background:linear-gradient(135deg,#16A34A,#22C55E);
        color:white;
        border:none;
        padding:12px 20px;
        border-radius:12px;
        cursor:pointer;
        font-weight:700;
        box-shadow:0 8px 20px rgba(34,197,94,0.25);
        font-family:Arial, sans-serif;">
        📎 Share PDF File
      </button>
      <p id="shareStatus" style="font-family:Arial, sans-serif; font-size:13px; color:#475569;"></p>
    </div>
    <script>
    const btn = document.getElementById('sharePdfBtn');
    const status = document.getElementById('shareStatus');
    btn.onclick = async () => {{
      try {{
        const b64 = "{pdf_b64}";
        const byteCharacters = atob(b64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {{
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }}
        const byteArray = new Uint8Array(byteNumbers);
        const file = new File([byteArray], "{file_name}", {{type: 'application/pdf'}});
        if (navigator.canShare && navigator.canShare({{ files: [file] }})) {{
          await navigator.share({{
            title: 'GlucoTrack Diabetes Report',
            text: `{safe_caption}`,
            files: [file]
          }});
          status.innerText = 'Share panel opened. Select WhatsApp to send the PDF.';
        }} else {{
          status.innerText = 'Your browser cannot directly share PDF files. Please download the report and attach it in WhatsApp.';
        }}
      }} catch (err) {{
        status.innerText = 'PDF sharing is not supported here. Download the report and attach it in WhatsApp.';
      }}
    }};
    </script>
    """, height=95)

def find_user_by_email_or_phone(identifier):
    """Find a user using registered email ID or phone number."""
    search_value = (identifier or "").strip().lower()
    search_phone = clean_phone(search_value)

    for username, user_data in st.session_state.users_db.items():
        email = str(user_data.get("email", "")).strip().lower()
        phone = clean_phone(str(user_data.get("phone", "")))

        if search_value and search_value == email:
            return username, user_data

        if search_phone and search_phone == phone:
            return username, user_data

    return None, None

def get_current_user_data():
    if st.session_state.current_user:
        return st.session_state.users_db.get(st.session_state.current_user, {})
    return {}

def create_chart_image(title, labels, values):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("Value")
    plt.tight_layout()

    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format="png")
    img_buffer.seek(0)
    plt.close(fig)
    return img_buffer

def generate_patient_report(patient_name, patient_id, result_text, patient_data, patient_photo=None, include_photo_slot=False, report_for="Patient"):
    """Create a professional PDF report with styled sections, tables and charts."""
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in (patient_name or "patient"))
    file_path = os.path.join(tempfile.gettempdir(), f"{safe_name}_diabetes_report.pdf")

    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4

    primary = colors.HexColor("#1E3A8A")
    secondary = colors.HexColor("#2563EB")
    accent = colors.HexColor("#7C3AED")
    light_bg = colors.HexColor("#F8FAFC")
    border = colors.HexColor("#D8DEE9")
    text_dark = colors.HexColor("#111827")
    text_muted = colors.HexColor("#6B7280")
    success = colors.HexColor("#16A34A")
    danger = colors.HexColor("#DC2626")
    warning = colors.HexColor("#F59E0B")

    def footer(page_no):
        c.setStrokeColor(border)
        c.line(45, 42, width - 45, 42)
        c.setFillColor(text_muted)
        c.setFont("Helvetica", 8)
        c.drawString(50, 28, "Generated by GLUCOTRACK | Project/Demo Report")
        c.drawRightString(width - 50, 28, f"Page {page_no}")

    def header(title, subtitle=None):
        c.setFillColor(primary)
        c.rect(0, height - 95, width, 95, fill=True, stroke=False)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 22)
        c.drawString(50, height - 45, title)
        if subtitle:
            c.setFont("Helvetica", 10)
            c.drawString(50, height - 65, subtitle)
        c.setFillColor(secondary)
        c.roundRect(width - 165, height - 62, 115, 30, 10, fill=True, stroke=False)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 11)
        c.drawCentredString(width - 107, height - 51, "GLUCOTRACK")

    def section_title(y, title):
        c.setFillColor(primary)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(50, y, title)
        c.setStrokeColor(secondary)
        c.setLineWidth(2)
        c.line(50, y - 6, 180, y - 6)
        return y - 26

    def info_card(x, y, w, h, label, value, color=primary):
        c.setFillColor(light_bg)
        c.setStrokeColor(border)
        c.roundRect(x, y, w, h, 12, fill=True, stroke=True)
        c.setFillColor(text_muted)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(x + 14, y + h - 18, label.upper())
        c.setFillColor(color)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(x + 14, y + 16, str(value))

    generated_on = datetime.now().strftime('%d-%m-%Y %I:%M %p')
    glucose_value = float(patient_data["Glucose"].values[0])
    bmi_value = float(patient_data["BMI"].values[0])
    result_color = danger if "High" in result_text else success
    risk_label = "HIGH RISK" if "High" in result_text else "LOW RISK"

    # Page 1: Summary and clinical values
    header("Diabetes Prediction Report", f"Generated on {generated_on}")

    c.setFillColor(colors.white)
    c.setStrokeColor(border)
    c.roundRect(45, height - 210, width - 90, 85, 14, fill=True, stroke=True)

    c.setFillColor(text_dark)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(65, height - 150, "Patient Summary")
    c.setFont("Helvetica", 10)
    c.setFillColor(text_muted)
    c.drawString(65, height - 170, f"Patient Name: {patient_name}")
    if patient_id and str(patient_id) != "Self Check":
        c.drawString(65, height - 188, f"Patient ID: {patient_id}")
    else:
        c.drawString(65, height - 188, "Patient Type: Self Check")
    c.drawString(65, height - 206, f"Report Created By: {report_for}")

    c.setFillColor(result_color)
    c.roundRect(width - 205, height - 185, 130, 34, 10, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(width - 140, height - 173, risk_label)

    if include_photo_slot:
        c.setStrokeColor(border)
        c.setFillColor(light_bg)
        c.roundRect(width - 95, height - 207, 55, 55, 8, fill=True, stroke=True)
        if patient_photo:
            try:
                photo_bytes = base64.b64decode(patient_photo)
                photo_img = ImageReader(io.BytesIO(photo_bytes))
                c.drawImage(photo_img, width - 90, height - 202, 45, 45, mask='auto')
            except Exception:
                c.setFillColor(text_muted)
                c.setFont("Helvetica", 7)
                c.drawCentredString(width - 67, height - 178, "Photo")
        else:
            c.setFillColor(text_muted)
            c.setFont("Helvetica", 7)
            c.drawCentredString(width - 67, height - 178, "Photo")

    y = section_title(height - 245, "Key Health Indicators")
    info_card(50, y - 55, 115, 52, "Glucose", patient_data["Glucose"].values[0], danger if glucose_value >= 126 else primary)
    info_card(180, y - 55, 115, 52, "BMI", patient_data["BMI"].values[0], warning if bmi_value >= 25 else primary)
    info_card(310, y - 55, 115, 52, "Blood Pressure", patient_data["BloodPressure"].values[0], primary)
    info_card(440, y - 55, 105, 52, "Insulin", patient_data["Insulin"].values[0], primary)

    y = section_title(height - 360, "Clinical Input Values")
    table_x = 55
    table_y = y
    row_h = 24
    col_w1 = 245
    col_w2 = 210

    c.setFillColor(primary)
    c.roundRect(table_x, table_y - row_h, col_w1 + col_w2, row_h, 8, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(table_x + 12, table_y - 16, "Input Field")
    c.drawString(table_x + col_w1 + 12, table_y - 16, "Patient Value")

    y_cursor = table_y - row_h
    c.setFont("Helvetica", 10)
    for i, col in enumerate(patient_data.columns):
        y_cursor -= row_h
        c.setFillColor(colors.HexColor("#FFFFFF") if i % 2 == 0 else colors.HexColor("#F3F6FB"))
        c.rect(table_x, y_cursor, col_w1 + col_w2, row_h, fill=True, stroke=False)
        c.setStrokeColor(border)
        c.rect(table_x, y_cursor, col_w1 + col_w2, row_h, fill=False, stroke=True)
        c.setFillColor(text_dark)
        c.drawString(table_x + 12, y_cursor + 8, str(col))
        c.drawString(table_x + col_w1 + 12, y_cursor + 8, str(patient_data[col].values[0]))

    c.setFillColor(text_muted)
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(50, 72, "Note: This prediction is generated by a machine learning model and should be verified by a qualified medical professional.")
    footer(1)
    c.showPage()

    # Page 2: Visual report
    header("Patient Data Visualization", "Visual summary of important clinical values")

    chart1 = create_chart_image(
        "Metabolic Indicators",
        ["Glucose", "BMI", "Age"],
        [patient_data["Glucose"].values[0], patient_data["BMI"].values[0], patient_data["Age"].values[0]]
    )
    chart2 = create_chart_image(
        "Blood Pressure, Insulin and Skin",
        ["BP", "Insulin", "Skin"],
        [patient_data["BloodPressure"].values[0], patient_data["Insulin"].values[0], patient_data["SkinThickness"].values[0]]
    )

    c.setFillColor(light_bg)
    c.setStrokeColor(border)
    c.roundRect(45, height - 360, width - 90, 240, 14, fill=True, stroke=True)
    c.drawImage(ImageReader(chart1), 70, height - 335, 455, 190)

    c.setFillColor(light_bg)
    c.setStrokeColor(border)
    c.roundRect(45, height - 625, width - 90, 240, 14, fill=True, stroke=True)
    c.drawImage(ImageReader(chart2), 70, height - 600, 455, 190)

    footer(2)
    c.showPage()

    # Page 3: Advice
    header("Health Advice", "General guidance based on entered values")

    if "High" in result_text:
        advice_title = "Recommended Action: Medical Consultation Needed"
        advice_color = danger
        advice_lines = [
            "Consult a doctor or diabetes specialist as soon as possible.",
            "Avoid excess sugar, sweet drinks, fried food and junk food.",
            "Walk for at least 30 minutes daily, if medically suitable.",
            "Follow a balanced diet with vegetables, protein and fiber-rich food.",
            "Monitor fasting and post-meal glucose regularly."
        ]
    elif 100 <= glucose_value < 126:
        advice_title = "Recommended Action: Lifestyle Improvement"
        advice_color = warning
        advice_lines = [
            "The glucose value may be in the prediabetes range.",
            "Reduce refined carbohydrates and sugary food items.",
            "Maintain a healthy body weight.",
            "Do regular physical exercise.",
            "Repeat glucose testing after medical consultation."
        ]
    else:
        advice_title = "Recommended Action: Maintain Healthy Routine"
        advice_color = success
        advice_lines = [
            "Current risk appears low.",
            "Continue a healthy lifestyle and balanced diet.",
            "Maintain BMI in the normal range.",
            "Avoid excess sugar and processed food.",
            "Go for routine health checkups."
        ]

    c.setFillColor(advice_color)
    c.roundRect(50, height - 165, width - 100, 48, 12, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(70, height - 145, advice_title)

    y = height - 220
    c.setFont("Helvetica", 11)
    for i, line in enumerate(advice_lines, 1):
        c.setFillColor(light_bg)
        c.setStrokeColor(border)
        c.roundRect(60, y - 10, width - 120, 32, 8, fill=True, stroke=True)
        c.setFillColor(primary)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(78, y, f"{i}.")
        c.setFillColor(text_dark)
        c.setFont("Helvetica", 11)
        c.drawString(105, y, line)
        y -= 42

    c.setFillColor(text_muted)
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 75, "Disclaimer: This report is for project/demo purpose only and is not a replacement for medical diagnosis.")
    footer(3)

    c.save()
    return file_path


def get_medical_advice(glucose, bmi, bp, insulin, prediction_value):
    advice = []
    status = "Healthy Advice"
    level = "success"
    if prediction_value == 1:
        status = "High Risk Advice"
        level = "warning"
        advice.append("The model shows high diabetes risk. The patient should consult a doctor or diabetologist for confirmatory testing.")
    if glucose >= 126:
        status = "High Glucose Advice"
        level = "warning"
        advice.append("Glucose is in a high range. Reduce sweet drinks, sweets, refined carbs and monitor fasting/post-meal glucose.")
    elif 100 <= glucose < 126:
        status = "Prediabetes Advice"
        level = "info"
        advice.append("Glucose is in the prediabetes range. Improve diet, exercise regularly and repeat glucose testing after medical guidance.")
    else:
        advice.append("Glucose is currently in a safer range. Continue routine monitoring and a balanced lifestyle.")

    if bmi >= 30:
        advice.append("BMI is in the obesity range. A structured weight management plan and daily physical activity are recommended.")
    elif bmi >= 25:
        advice.append("BMI is in the overweight range. Focus on portion control, walking and reducing processed food.")

    if bp >= 90:
        advice.append("Blood pressure value is high. Salt intake should be controlled and blood pressure should be checked regularly.")

    if insulin > 200:
        advice.append("Insulin value is high. The doctor should review insulin resistance risk and related metabolic markers.")

    if not advice:
        advice.append("Maintain a healthy diet, regular exercise, good sleep and yearly health checkups.")
    return status, level, advice

def show_medical_advice(glucose, bmi, bp, insulin, prediction_value):
    status, level, advice = get_medical_advice(glucose, bmi, bp, insulin, prediction_value)
    message = "**" + status + ":**\n" + "\n".join([f"- {item}" for item in advice])
    if level == "warning":
        st.warning(message)
    elif level == "info":
        st.info(message)
    else:
        st.success(message)


def get_risk_reasons(glucose, bmi, bp, insulin, prediction_value):
    reasons = []
    if glucose >= 126:
        reasons.append("Glucose is in the diabetes-risk range, so it is a major reason for high risk.")
    elif glucose >= 100:
        reasons.append("Glucose is in the prediabetes range, so lifestyle correction is important.")
    if bmi >= 30:
        reasons.append("BMI is in the obesity range, which can increase insulin resistance.")
    elif bmi >= 25:
        reasons.append("BMI is in the overweight range, which can increase diabetes risk.")
    if bp >= 90:
        reasons.append("Blood pressure is high and should be monitored with medical advice.")
    if insulin > 200:
        reasons.append("Insulin value is high, which may indicate insulin resistance and needs clinical review.")
    if prediction_value == 1 and not reasons:
        reasons.append("The model detected high risk from the combined pattern of all medical inputs.")
    if not reasons:
        reasons.append("No major high-risk trigger is visible from these values, but routine monitoring is advised.")
    return reasons

def save_current_patient_values(current_user, preg, glucose, bp, skin, insulin, bmi, dpf, age, prediction=None, probability=None):
    input_raw = pd.DataFrame({
        'Pregnancies': [preg],
        'Glucose': [glucose],
        'BloodPressure': [bp],
        'SkinThickness': [skin],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })
    input_raw['Glucose_BMI'] = input_raw['Glucose'] * input_raw['BMI']
    input_raw['Insulin_Glucose'] = input_raw['Insulin'] * input_raw['Glucose']
    input_raw['Age_BMI'] = input_raw['Age'] * input_raw['BMI']
    input_raw['BMI_Squared'] = input_raw['BMI'] ** 2
    st.session_state.input_raw = input_raw
    role = current_user.get("role", "Patient")
    now = datetime.now().strftime("%d-%m-%Y %I:%M %p")
    if role == "Doctor" and st.session_state.active_patient_id:
        doctor_key = st.session_state.current_user
        st.session_state.patients_db.setdefault(doctor_key, {})
        st.session_state.patients_db[doctor_key].setdefault(st.session_state.active_patient_id, {})
        st.session_state.patients_db[doctor_key][st.session_state.active_patient_id].update({
            "name": st.session_state.active_patient_name,
            "patient_id": st.session_state.active_patient_id,
            "age": age,
            "gender": st.session_state.active_patient_gender,
            "photo": st.session_state.patient_photo,
            "last_input": input_raw.to_dict("records")[0],
            "last_prediction": None if prediction is None else int(prediction),
            "last_probability": probability,
            "last_checked": now,
        })
        save_local_data()
        return True, "Patient data saved successfully."
    elif role == "Patient" and st.session_state.current_user:
        st.session_state.users_db[st.session_state.current_user].update({
            "full_name": st.session_state.active_patient_name or current_user.get("full_name", ""),
            "age": age,
            "gender": st.session_state.active_patient_gender,
            "last_input": input_raw.to_dict("records")[0],
            "last_prediction": None if prediction is None else int(prediction),
            "last_probability": probability,
            "last_checked": now,
        })
        save_local_data()
        return True, "Your data has been saved."
    return False, "Please select or enroll a patient before saving."

# ==============================
# SIDEBAR / NAVIGATION
# ==============================
# On the first Welcome screen, the sidebar navigation is hidden.
# After clicking Get Started, the user goes to Login and the navigation appears.
show_sidebar = not (
    st.session_state.selected_menu == "🏠 Welcome"
    and not st.session_state.get("logged_in", False)
)

if show_sidebar:
    st.sidebar.markdown("""
    <div style='padding: 0.6rem 0 1rem 0;'>
      <span style='font-size:1.5rem; font-weight:800;'>🩺 GLUCOTRACK</span><br>
      <span style='font-size:0.75rem; opacity:0.55; letter-spacing:0.08em;'>
          SMART HEALTH DASHBOARD
      </span>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get("logged_in", False):
        current_user_data = st.session_state.users_db.get(st.session_state.current_user, {})
        current_role = current_user_data.get("role", "Patient")

        if current_role == "Doctor":
            menu_options = [
                "👋 Doctor Home",
                "📋 Enroll Patient",
                "👥 Patient Details",
                "🔬 Prediction",
                "📊 Visualization",
                "ℹ️ About"
            ]
        else:
            menu_options = [
                "🔬 Prediction",
                "📊 Visualization",
                "ℹ️ About"
            ]
    else:
        menu_options = [
            "🔐 Login",
            "📝 Sign Up",
            "ℹ️ About"
        ]

    if st.session_state.selected_menu not in menu_options:
        st.session_state.selected_menu = menu_options[0]

    default_index = menu_options.index(st.session_state.selected_menu)
    menu = st.sidebar.radio("Navigation", menu_options, index=default_index)
    if menu != st.session_state.selected_menu:
        st.session_state.previous_menu = st.session_state.selected_menu
        history = st.session_state.get("page_history", [])
        if not history or history[-1] != st.session_state.selected_menu:
            history.append(st.session_state.selected_menu)
        st.session_state.page_history = history[-12:]
        st.session_state.selected_menu = menu
        try:
            st.query_params["page"] = PAGE_SLUGS.get(menu, "")
            if st.session_state.get("current_user"):
                st.query_params["user"] = st.session_state.current_user
        except Exception:
            pass
        st.rerun()

    st.sidebar.divider()

    dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=(st.session_state.theme == "Dark"))
    st.session_state.theme = "Dark" if dark_mode else "Light"
else:
    menu = "🏠 Welcome"
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    div[data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    .welcome-hero-center {{
        max-width: 980px;
        margin: 0 auto;
        text-align: center;
        padding: 2.5rem 1rem 1rem 1rem;
    }}
    .welcome-text-area {{
        max-width: 760px;
        margin: 0 auto 1.5rem auto;
    }}
    .welcome-image-card {{
        max-width: 760px;
        height: 330px;
        margin: 1.6rem auto 0 auto;
        border-radius: 26px;
        overflow: hidden;
        border: 1px solid {border};
        box-shadow: 0 24px 70px rgba(15,23,42,0.16);
        background: linear-gradient(135deg, {accent}, {accent2});
    }}
    .welcome-image-card img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
        opacity: 0.94;
    }}

    </style>
    """, unsafe_allow_html=True)

apply_theme(st.session_state.theme)
show_professional_back_button()

# ==============================
# WELCOME PAGE
# ==============================
if menu == "🏠 Welcome":
    st.markdown("""
    <div class="welcome-hero-center">
      <div class="welcome-text-area">
        <div class="brand-pill">🩺 GLUCOTRACK</div>
        <div class="hero-title">Smarter Diabetes Risk Screening</div>
        <div class="hero-sub">A professional machine learning dashboard for quick diabetes risk prediction, patient visualization and report generation.</div>
      </div>
      <div class="welcome-image-card">
        <img src="https://images.unsplash.com/photo-1576091160550-2173dba999ef?auto=format&fit=crop&w=900&q=80" alt="Diabetes healthcare" />
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 1, 2])
    with c2:
        if st.button("Get Started →", use_container_width=True):
            go_to_page("🔐 Login")

# ==============================
# LOGIN PAGE
# ==============================
elif menu == "🔐 Login":
    st.markdown("""
<div style='text-align:center; padding:2.2rem 1rem 1rem 1rem; margin-bottom:1rem;'>

<div style='
    display:inline-block;
    padding:0.45rem 1rem;
    border-radius:999px;
    background:linear-gradient(135deg, rgba(14,165,233,0.18), rgba(45,212,191,0.18));
    color:#0EA5E9;
    font-size:0.82rem;
    font-weight:700;
    letter-spacing:0.08em;
    margin-bottom:1rem;
'>
🔐 SECURE LOGIN
</div>

<h1 style='
    font-size:3rem;
    margin-bottom:0.5rem;
    font-weight:800;
'>
Welcome Back
</h1>

<p style='
    font-size:1rem;
    opacity:0.8;
    max-width:620px;
    margin:auto;
    line-height:1.7;
'>
Access your GlucoTrack dashboard securely and continue managing diabetes prediction, patient records and reports.
</p>

</div>
""", unsafe_allow_html=True)

    left_space, login_box, right_space = st.columns([1, 1.3, 1])

    with login_box:
        login_user = st.text_input("Username", placeholder="Enter your username", key="password_login_user")
        login_password = st.text_input("Password", type="password", placeholder="Enter your password", key="password_login_pass")

        if st.button("Login →", use_container_width=True):
            db = st.session_state.users_db
            if login_user in db and str(db[login_user].get("password", "")) == login_password:
                user_data = db[login_user]
                st.session_state.logged_in = True
                st.session_state.current_user = login_user
                st.query_params["user"] = login_user

                if user_data.get("role") == "Doctor":
                    st.session_state.selected_menu = "👋 Doctor Home"
                else:
                    st.session_state.active_patient_name = user_data.get("full_name", "")
                    st.session_state.active_patient_id = user_data.get("patient_id", "")
                    st.session_state.active_patient_age = user_data.get("age")
                    st.session_state.active_patient_gender = user_data.get("gender", "")
                    st.session_state.patient_photo = None
                    st.session_state.selected_menu = "🔬 Prediction"

                try:
                    st.query_params["page"] = PAGE_SLUGS.get(st.session_state.selected_menu, "")
                except Exception:
                    pass
                st.success(f"Welcome, {user_data.get('full_name', 'User')}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

# ==============================
# SIGN UP PAGE
# ==============================
elif menu == "📝 Sign Up":
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.markdown("## Create Account")
    st.markdown("Create a Doctor or Patient account. After sign up, the correct page will open automatically.")
    st.markdown("<br>", unsafe_allow_html=True)

    col_l = st.container()

    with col_l:

        account_type = st.radio("Account Type", ["Patient", "Doctor"], horizontal=True)

        st.markdown("#### Personal Details")
        su_fullname = st.text_input("Full Name", placeholder="Enter full name", key="su_name")

        if account_type == "Doctor":
            su_specialization = st.text_input("Specialization", placeholder="Example: General Physician")
            su_license = st.text_input("Medical License ID", placeholder="Optional")
            su_age = None
            su_gender = None
            su_patient_id = None
        else:
            # A self-check patient does not need a Patient ID.
            su_patient_id = ""
            su_age = st.number_input("Age", 1, 120, 25, key="su_age")
            su_gender = st.selectbox("Gender", ["Female", "Male", "Other"], key="su_gender")
            su_specialization = None
            su_license = None

        st.markdown("#### Contact Details")
        su_email = st.text_input("Email ID", placeholder="Example: rose@gmail.com", key="su_email")
        su_phone = st.text_input("Phone Number", placeholder="Example: 9876543210", key="su_phone")

        st.markdown("#### Login Details")
        su_username = st.text_input("Choose Username", placeholder="Example: rose123", key="su_user")
        su_password = st.text_input("Create Password", type="password", placeholder="Minimum 4 characters", key="su_pass")
        su_password2 = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="su_pass2")

        if st.button("Create Account", use_container_width=True):
            db = st.session_state.users_db

            if not su_fullname or not su_username or not su_email or not su_phone or not su_password or not su_password2:
                st.warning("Please fill full name, email ID, phone number, username and password.")
            elif "@" not in su_email or "." not in su_email:
                st.error("Please enter a valid email ID.")
            elif len(clean_phone(su_phone)) < 10:
                st.error("Please enter a valid phone number.")
            elif len(su_username) < 3:
                st.error("Username must be at least 3 characters.")
            elif " " in su_username:
                st.error("Username cannot contain spaces.")
            elif su_username in db:
                st.error("Username already exists. Choose another username.")
            elif len(su_password) < 4:
                st.error("Password must be at least 4 characters.")
            elif su_password != su_password2:
                st.error("Passwords do not match.")
            elif any(str(user.get("email", "")).strip().lower() == su_email.strip().lower() for user in db.values()):
                st.error("This email ID is already registered.")
            elif any(clean_phone(str(user.get("phone", ""))) == clean_phone(su_phone) for user in db.values()):
                st.error("This phone number is already registered.")
            else:
                db[su_username] = {
                    "password": su_password,
                    "full_name": su_fullname,
                    "role": account_type,
                    "account_type": account_type,
                    "email": su_email.strip(),
                    "phone": clean_phone(su_phone),
                    "specialization": su_specialization,
                    "license": su_license,
                    "age": su_age,
                    "gender": su_gender,
                    "patient_id": su_patient_id
                }

                st.session_state.users_db = db
                save_local_data()
                st.session_state.logged_in = True
                st.session_state.current_user = su_username
                st.query_params["user"] = su_username

                if account_type == "Doctor":
                    st.session_state.selected_menu = "📋 Enroll Patient"
                else:
                    st.session_state.active_patient_name = su_fullname
                    st.session_state.active_patient_id = ""
                    st.session_state.patient_photo = None
                    st.session_state.active_patient_age = su_age
                    st.session_state.active_patient_gender = su_gender
                    st.session_state.selected_menu = "🔬 Prediction"

                try:
                    st.query_params["page"] = PAGE_SLUGS.get(st.session_state.selected_menu, "")
                except Exception:
                    pass
                st.success(f"{account_type} account created successfully.")
                st.balloons()
                st.rerun()

# ==============================
# DOCTOR HOME PAGE
# ==============================
elif menu == "👋 Doctor Home":
    if not st.session_state.logged_in:
        st.warning("Please login first.")
    else:
        current = get_current_user_data()
        if current.get("role") != "Doctor" and current.get("role") != "Admin":
            st.warning("This page is for doctors only.")
        else:
            doctor_name = current.get("full_name", "Doctor")
            doctor_key = st.session_state.current_user
            total_patients = len(st.session_state.patients_db.get(doctor_key, {}))
            st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
            st.markdown(f"## Welcome, Dr. {doctor_name}")
            st.markdown("Manage patient enrollment, prediction, saved reports and follow-up checks from one place.")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="responsive-card"><h3>{total_patients}</h3><p>Saved Patients</p></div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="responsive-card"><h3>ML</h3><p>Diabetes Risk Prediction</p></div>', unsafe_allow_html=True)
            with c3:
                st.markdown('<div class="responsive-card"><h3>PDF</h3><p>Professional Reports</p></div>', unsafe_allow_html=True)
            st.markdown("---")
            a, b = st.columns(2)
            with a:
                if st.button("➕ Enroll New Patient", use_container_width=True):
                    go_to_page("📋 Enroll Patient")
            with b:
                if st.button("👥 View Patient Details", use_container_width=True):
                    go_to_page("👥 Patient Details")

# ==============================
# PATIENT DETAILS PAGE
# ==============================
elif menu == "👥 Patient Details":
    if not st.session_state.logged_in:
        st.warning("Please login first.")
    else:
        current = get_current_user_data()
        if current.get("role") != "Doctor" and current.get("role") != "Admin":
            st.warning("Only doctors can view enrolled patient details.")
        else:
            doctor_key = st.session_state.current_user
            doctor_patients = st.session_state.patients_db.get(doctor_key, {})
            st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
            st.markdown("## Patient Details")
            st.markdown("These are the patients enrolled by the logged-in doctor only.")

            if not doctor_patients:
                st.info("No patients enrolled yet. Add a patient from the Enroll Patient page.")
                if st.button("Enroll New Patient"):
                    go_to_page("📋 Enroll Patient")
            else:
                search_text = st.text_input("Search by Patient ID or Name", placeholder="Example: PT-001 or Ramesh")
                rows = []
                for pid, pdata in doctor_patients.items():
                    if search_text.strip():
                        q = search_text.strip().lower()
                        if q not in pid.lower() and q not in pdata.get("name", "").lower():
                            continue
                    rows.append({
                        "Patient ID": pdata.get("patient_id", pid),
                        "Name": pdata.get("name", ""),
                        "Age": pdata.get("age", ""),
                        "Gender": pdata.get("gender", ""),
                        "Contact": pdata.get("contact", ""),
                        "Last Glucose": pdata.get("last_input", {}).get("Glucose", ""),
                        "Last BMI": pdata.get("last_input", {}).get("BMI", ""),
                        "Last Checked": pdata.get("last_checked", ""),
                        "Notes": pdata.get("notes", "")
                    })

                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    selected_id = st.selectbox("Select Patient ID to open", [r["Patient ID"] for r in rows])
                    open_col, delete_col = st.columns(2)
                    with open_col:
                        open_selected = st.button("Open Selected Patient in Prediction", use_container_width=True)
                    with delete_col:
                        delete_selected = st.button("🗑️ Delete Selected Patient", use_container_width=True)
                    if delete_selected:
                        if selected_id in st.session_state.patients_db.get(doctor_key, {}):
                            del st.session_state.patients_db[doctor_key][selected_id]
                            save_local_data()
                            st.success("Patient record deleted successfully.")
                            st.rerun()
                    if open_selected:
                        saved_patient = doctor_patients.get(selected_id)
                        if saved_patient:
                            st.session_state.active_patient_name = saved_patient.get("name", "")
                            st.session_state.active_patient_id = saved_patient.get("patient_id", "")
                            st.session_state.active_patient_age = saved_patient.get("age")
                            st.session_state.active_patient_gender = saved_patient.get("gender", "")
                            st.session_state.patient_photo = saved_patient.get("photo")
                            last_input = saved_patient.get("last_input")
                            if last_input:
                                st.session_state.input_raw = pd.DataFrame([last_input])
                                st.session_state.prediction = saved_patient.get("last_prediction")
                                st.session_state.probability = saved_patient.get("last_probability")
                            else:
                                for clear_key in ["input_raw", "prediction", "probability"]:
                                    if clear_key in st.session_state:
                                        del st.session_state[clear_key]
                            for widget_key in ["pred_preg", "pred_glucose", "pred_bp", "pred_skin", "pred_insulin", "pred_bmi", "pred_dpf", "pred_age"]:
                                if widget_key in st.session_state:
                                    del st.session_state[widget_key]
                            go_to_page("🔬 Prediction")
                else:
                    st.warning("No matching patient found.")

# ==============================
# ENROLL PATIENT PAGE
# ==============================
elif menu == "📋 Enroll Patient":
    if not st.session_state.logged_in:
        st.warning("Please login first.")
    else:
        current = get_current_user_data()
        if current.get("role") != "Doctor" and current.get("role") != "Admin":
            st.warning("Only doctors or admins can access the Patient Enrollment page.")
        else:
            st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
            st.markdown("## Patient Enrollment")
            st.markdown("Create a patient record before running the diabetes prediction.")
            st.markdown("<br>", unsafe_allow_html=True)

            doctor_key = st.session_state.current_user
            if doctor_key not in st.session_state.patients_db:
                st.session_state.patients_db[doctor_key] = {}

            st.markdown("### Enroll New Patient")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("#### Patient Details")
                p_name = st.text_input("Patient Full Name", placeholder="Example: Ramesh Kumar", key="enroll_name")
                p_id = st.text_input("Patient ID", placeholder="Example: PT-20260001", key="enroll_id")
                p_age = st.number_input("Age", 1, 120, 35, key="enroll_age")
                p_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="enroll_gender")
                p_contact = st.text_input("Contact Number", placeholder="+91 XXXXX XXXXX", key="enroll_contact")
                p_addr = st.text_area("Address", placeholder="City, State", height=70, key="enroll_address")

            with col2:
                st.markdown("#### Patient Photo")
                photo = st.file_uploader("Upload Photo (JPG / PNG)", type=["jpg", "jpeg", "png"])

                uploaded_photo_b64 = None
                if photo:
                    img = Image.open(photo)
                    w, h = img.size
                    side = min(w, h)
                    left = (w - side) // 2
                    top = (h - side) // 2
                    img_cropped = img.crop((left, top, left + side, top + side))
                    img_resized = img_cropped.resize((240, 240))

                    buf = io.BytesIO()
                    img_resized.save(buf, format="PNG")
                    uploaded_photo_b64 = base64.b64encode(buf.getvalue()).decode()

                    st.markdown(f"""
                    <div style='text-align:center; margin-top:0.5rem;'>
                      <img src="data:image/png;base64,{uploaded_photo_b64}"
                           style="border-radius:14px; width:200px; height:200px;
                           object-fit:cover; border:3px solid #0F766E;
                           box-shadow:0 4px 20px rgba(15,118,110,0.35);" />
                      <div style='font-size:0.75rem; margin-top:0.5rem; opacity:0.6;'>Photo Preview</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No photo uploaded.")

                p_notes = st.text_area("Medical Notes", height=80, placeholder="Optional notes", key="enroll_notes")

            if st.button("Enroll Patient and Open Prediction"):
                clean_patient_id = p_id.strip()
                if not p_name or not clean_patient_id:
                    st.warning("Please enter Patient Name and Patient ID.")
                elif clean_patient_id in st.session_state.patients_db.get(doctor_key, {}):
                    st.error("This Patient ID already exists for this doctor. Please use a unique Patient ID or open the existing patient by ID.")
                else:
                    patient_record = {
                        "name": p_name,
                        "patient_id": clean_patient_id,
                        "age": p_age,
                        "gender": p_gender,
                        "contact": p_contact,
                        "address": p_addr,
                        "notes": p_notes,
                        "photo": uploaded_photo_b64,
                    }
                    st.session_state.patients_db[doctor_key][clean_patient_id] = patient_record
                    save_local_data()
                    st.session_state.active_patient_name = p_name
                    st.session_state.active_patient_id = clean_patient_id
                    st.session_state.active_patient_age = p_age
                    st.session_state.active_patient_gender = p_gender
                    st.session_state.patient_photo = uploaded_photo_b64
                    for widget_key in ["pred_preg", "pred_glucose", "pred_bp", "pred_skin", "pred_insulin", "pred_bmi", "pred_dpf", "pred_age"]:
                        if widget_key in st.session_state:
                            del st.session_state[widget_key]
                    for clear_key in ["input_raw", "prediction", "probability"]:
                        if clear_key in st.session_state:
                            del st.session_state[clear_key]
                    st.session_state.selected_menu = "🔬 Prediction"
                    st.success(f"Patient {p_name} enrolled successfully.")
                    st.rerun()

# ==============================
# PREDICTION PAGE
# ==============================
elif menu == "🔬 Prediction":
    if not st.session_state.logged_in:
        st.warning("Please login first.")
    else:
        current_user = get_current_user_data()

        current_role = current_user.get("role", "Patient")

        if current_role == "Patient":
            st.session_state.active_patient_name = st.session_state.active_patient_name or current_user.get("full_name", "")
            st.session_state.active_patient_id = ""
            st.session_state.patient_photo = None
            st.session_state.active_patient_age = current_user.get("age")
            st.session_state.active_patient_gender = current_user.get("gender", "")
            if "input_raw" not in st.session_state and current_user.get("last_input"):
                st.session_state.input_raw = pd.DataFrame([current_user.get("last_input")])
                st.session_state.prediction = current_user.get("last_prediction")
                st.session_state.probability = current_user.get("last_probability")

        saved_defaults = {}
        if "input_raw" in st.session_state:
            try:
                saved_defaults = st.session_state.input_raw.to_dict("records")[0]
            except Exception:
                saved_defaults = {}

        st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

        header_col, photo_col = st.columns([4, 1])

        with header_col:
            st.markdown("## Diabetes Risk Assessment")
            st.markdown("Enter the patient's health values, save the data, then run prediction.")
            patient_name = st.text_input(
                "Patient Name",
                value=st.session_state.active_patient_name or current_user.get("full_name", ""),
                placeholder="Enter patient name"
            )
            if current_role == "Doctor":
                patient_id = st.text_input(
                    "Patient ID",
                    value=st.session_state.active_patient_id or "",
                    placeholder="Enter patient ID"
                )
                st.session_state.active_patient_id = patient_id
            else:
                patient_id = ""
                st.caption("Self-check mode: Patient ID is not required.")

            st.session_state.active_patient_name = patient_name

        with photo_col:
            if st.session_state.patient_photo:
                st.markdown(f"""
                <div style='text-align:right;'>
                  <img src="data:image/png;base64,{st.session_state.patient_photo}"
                       style="border-radius:12px; width:72px; height:72px;
                       object-fit:cover; border:2px solid #0F766E;" />
                </div>
                """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Basic Health Values")
            preg = st.number_input("Pregnancies", 0, 20, int(saved_defaults.get("Pregnancies", 1)), key="pred_preg")
            glucose = st.number_input("Glucose (mg/dL)", 50, 200, int(saved_defaults.get("Glucose", 120)), key="pred_glucose")
            bp = st.number_input("Blood Pressure (mm Hg)", 30, 120, int(saved_defaults.get("BloodPressure", 70)), key="pred_bp")
            skin = st.number_input("Skin Thickness (mm)", 0, 100, int(saved_defaults.get("SkinThickness", 20)), key="pred_skin")
    
        with col2:
            st.markdown("##### Body and Family Health Values")
            insulin = st.number_input("Insulin (μU/mL)", 0, 300, int(saved_defaults.get("Insulin", 100)), key="pred_insulin")
            bmi = st.number_input("BMI (kg/m²)", 10.0, 60.0, float(saved_defaults.get("BMI", 25.0)), key="pred_bmi")
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, float(saved_defaults.get("DiabetesPedigreeFunction", 0.5)), key="pred_dpf")
            default_age = int(saved_defaults.get("Age", st.session_state.active_patient_age if st.session_state.active_patient_age else 30))
            age = st.number_input("Age (years)", 1, 100, default_age, key="pred_age")
    
        save_col, predict_col = st.columns(2)
        with save_col:
            if st.button("💾 Save Patient Data", use_container_width=True):
                ok, msg = save_current_patient_values(current_user, preg, glucose, bp, skin, insulin, bmi, dpf, age)
                if ok:
                    st.success(msg)
                else:
                    st.warning(msg)
        with predict_col:
            run_prediction = st.button("Run Prediction", use_container_width=True)

        if run_prediction:
            ok, msg = save_current_patient_values(current_user, preg, glucose, bp, skin, insulin, bmi, dpf, age)
            input_raw = st.session_state.input_raw

            input_encoded = pd.get_dummies(input_raw)
            input_df = input_encoded.reindex(columns=columns, fill_value=0)

            prediction = model.predict(input_df)

            try:
                probability = model.predict_proba(input_df)[0][1]
            except Exception:
                probability = None

            st.session_state.prediction = prediction[0]
            st.session_state.probability = probability
            save_current_patient_values(current_user, preg, glucose, bp, skin, insulin, bmi, dpf, age, prediction[0], probability)

            go_to_page("📊 Visualization")

        if False and "prediction" in st.session_state and "input_raw" in st.session_state:
            input_raw = st.session_state.input_raw
            prediction_value = st.session_state.prediction
            probability = st.session_state.probability

            st.markdown("### Prediction Result")
            st.markdown(f"**Patient Name:** {st.session_state.active_patient_name or 'Not provided'}")
            if current_user.get("role") == "Doctor":
                st.markdown(f"**Patient ID:** {st.session_state.active_patient_id or 'Not provided'}")
            else:
                st.markdown("**Mode:** Self-check patient report")

            if prediction_value == 1:
                result_text = "High Risk of Diabetes"
                st.error("High Risk of Diabetes — Please consult a doctor.")
                st.markdown('<span class="badge badge-red">HIGH RISK</span>', unsafe_allow_html=True)
            else:
                result_text = "Low Risk of Diabetes"
                st.success("Low Risk of Diabetes.")
                st.markdown('<span class="badge badge-green">LOW RISK</span>', unsafe_allow_html=True)

            if probability is not None:
                st.metric("Diabetes Risk Probability", f"{probability * 100:.2f}%")

            st.markdown("### Patient Health Visualization")

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Glucose", "BMI", "Blood Pressure", "Insulin", "Overall"
            ])

            with tab1:
                st.markdown("#### Glucose Level")
                fig, ax = plt.subplots(figsize=(4.8, 2.3))
                ax.barh(["Patient"], [glucose], height=0.35)
                ax.axvspan(50, 99, alpha=0.08, label="Normal")
                ax.axvspan(100, 125, alpha=0.08, label="Prediabetes")
                ax.axvspan(126, 200, alpha=0.08, label="High")
                ax.axvline(100, linestyle="--", linewidth=1)
                ax.axvline(126, linestyle="--", linewidth=1)
                ax.set_xlim(50, 200)
                ax.set_xlabel("mg/dL")
                ax.set_title("Glucose Status")
                ax.legend(fontsize=7, loc="upper right")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=False)
                st.caption("Shows the patient glucose value against normal, prediabetes and high ranges.")

            with tab2:
                st.markdown("#### BMI Level")
                fig, ax = plt.subplots(figsize=(4.8, 2.5))
                categories = ["Under", "Normal", "Over", "Obese"]
                limits = [18.5, 24.9, 29.9, 40]
                ax.plot(categories, limits, marker="o", linewidth=2, label="Category limit")
                patient_category = "Under" if bmi < 18.5 else "Normal" if bmi < 25 else "Over" if bmi < 30 else "Obese"
                ax.scatter([patient_category], [bmi], s=90, marker="D", label="Patient BMI")
                ax.set_ylabel("BMI")
                ax.set_title("BMI Category Position")
                ax.grid(True, alpha=0.25)
                ax.legend(fontsize=7)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=False)
                st.caption("Shows the patient's BMI position in standard BMI categories.")

            with tab3:
                st.markdown("#### Blood Pressure Donut Chart")
                low_part = max(min(bp, 80), 0)
                high_part = max(120 - bp, 0)
                patient_part = max(120 - low_part - high_part, 1)
                values = [low_part, patient_part, high_part]
                labels = ["Lower Zone", "Patient Zone", "Remaining Zone"]
                fig, ax = plt.subplots(figsize=(3.8, 3.0))
                wedges, texts, autotexts = ax.pie(
                    values,
                    labels=labels,
                    autopct="%1.0f%%",
                    startangle=90,
                    pctdistance=0.78,
                    wedgeprops={"width": 0.38, "edgecolor": "white"},
                    textprops={"fontsize": 8}
                )
                ax.text(0, 0, f"{bp}\nmm Hg", ha="center", va="center", fontsize=12, fontweight="bold")
                ax.set_title("Blood Pressure Status", fontsize=11)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=False)
                st.caption("A cleaner donut chart showing blood pressure as a compact professional visual.")

            with tab4:
                st.markdown("#### Insulin Level")
                fig, ax = plt.subplots(figsize=(4.8, 2.5))
                ax.fill_between([0, 1, 2], [0, insulin, 300], alpha=0.16)
                ax.plot(["Low", "Patient", "Upper"], [0, insulin, 300], marker="o", linewidth=2)
                ax.set_ylabel("μU/mL")
                ax.set_title("Insulin Range View")
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=False)
                st.caption("Shows the insulin value in a simple range-style trend view.")

            with tab5:
                st.markdown("#### Overall Patient Profile")
                values = [glucose, bp, skin, insulin, bmi, dpf, age]
                labels = ["Glucose", "BP", "Skin", "Insulin", "BMI", "DPF", "Age"]

                fig, ax = plt.subplots(figsize=(5.3, 2.8))
                ax.plot(labels, values, marker="o", linewidth=2)
                ax.fill_between(labels, values, alpha=0.12)
                ax.set_title("Overall Health Profile", fontsize=11)
                ax.set_ylabel("Values")
                ax.grid(True, alpha=0.25)
                plt.xticks(rotation=20)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=False)
                st.caption("Overall graph is kept, but made smaller and cleaner.")

            st.markdown("### Medical Advice")

            if glucose >= 126 or prediction_value == 1:
                st.warning("""
                **Diabetes Risk Advice:**
                - Consult a doctor or diabetes specialist.
                - Avoid excess sugar, sweet drinks and junk food.
                - Walk for at least 30 minutes daily.
                - Monitor glucose levels regularly.
                - Follow a balanced diet with vegetables, protein and fiber-rich food.
                """)
            elif 100 <= glucose < 126:
                st.info("""
                **Prediabetes Advice:**
                - Improve daily lifestyle habits.
                - Maintain a healthy body weight.
                - Exercise regularly.
                - Reduce refined sugar, white rice and processed food.
                - Repeat glucose testing after medical consultation.
                """)
            else:
                st.success("""
                **Healthy Advice:**
                - Continue a balanced diet.
                - Stay physically active.
                - Go for routine health checkups.
                - Monitor BMI and glucose level regularly.
                """)

            st.markdown("---")
            st.markdown("## Patient Report")

            patient_name_report = st.text_input(
                "Patient Name for Report",
                value=st.session_state.active_patient_name or "Unknown Patient"
            )
            if current_user.get("role") == "Doctor":
                patient_id_report = st.text_input(
                    "Patient ID for Report",
                    value=st.session_state.active_patient_id or ""
                )
            else:
                patient_id_report = "Self Check"

            if st.button("Generate Report"):
                report_path = generate_patient_report(
                    patient_name_report,
                    patient_id_report,
                    result_text,
                    input_raw[['Pregnancies', 'Glucose', 'BloodPressure',
                               'SkinThickness', 'Insulin', 'BMI',
                               'DiabetesPedigreeFunction', 'Age']],
                    st.session_state.patient_photo,
                    include_photo_slot=(st.session_state.users_db.get(st.session_state.current_user, {}).get("role") == "Doctor"),
                    report_for=st.session_state.users_db.get(st.session_state.current_user, {}).get("role", "Patient")
                )

                with open(report_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()

                st.success("Professional report generated successfully.")

                st.markdown("""
                <div class="card" style="border-left:5px solid #2563EB;">
                    <h4 style="margin-bottom:0.4rem;">Report Ready</h4>
                    <p style="margin-bottom:0.2rem;">Download the PDF report or share the PDF file from the browser share button.</p>
                    <p style="font-size:0.85rem; opacity:0.75;">The PDF file share button works best on mobile browsers. If desktop browser blocks file sharing, download the PDF and attach it in WhatsApp.</p>
                </div>
                """, unsafe_allow_html=True)

                pdf_file_name = f"{patient_name_report}_professional_report.pdf"

                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=pdf_file_name,
                    mime="application/pdf"
                )

                share_caption = (
                    f"GLUCOTRACK Diabetes Prediction Report\n"
                    f"Patient Name: {patient_name_report}\n"
                    f"Patient ID: {patient_id_report}\n"
                    f"Prediction Result: {result_text}"
                )
                build_whatsapp_file_share_button(pdf_bytes, pdf_file_name, share_caption)

                whatsapp_message = quote(
                    f"GLUCOTRACK Diabetes Prediction Report\n\n"
                    f"Patient Name: {patient_name_report}\n"
                    f"Patient ID: {patient_id_report}\n"
                    f"Prediction Result: {result_text}\n"
                    f"Glucose: {input_raw['Glucose'].values[0]} mg/dL\n"
                    f"BMI: {input_raw['BMI'].values[0]} kg/m²\n"
                    f"Blood Pressure: {input_raw['BloodPressure'].values[0]} mm Hg\n\n"
                    f"PDF report has been generated. Please attach the downloaded PDF if required."
                )

                whatsapp_url = f"https://wa.me/?text={whatsapp_message}"

                st.markdown(
                    f"""
                    <a href="{whatsapp_url}" target="_blank" style="text-decoration:none;">
                        <button style="
                            background:linear-gradient(135deg,#16A34A,#22C55E);
                            color:white;
                            border:none;
                            padding:12px 20px;
                            border-radius:12px;
                            cursor:pointer;
                            font-weight:700;
                            margin-top:10px;
                            box-shadow:0 8px 20px rgba(34,197,94,0.25);">
                            Share Report Summary on WhatsApp
                        </button>
                    </a>
                    """,
                    unsafe_allow_html=True
                )

# ==============================
# VISUALIZATION PAGE
# ==============================
elif menu == "📊 Visualization":
    if not st.session_state.logged_in:
        st.warning("Please login first.")
    elif "prediction" not in st.session_state or "input_raw" not in st.session_state:
        st.info("Please complete prediction first. Open the Prediction page and click Run Prediction.")
        if st.button("Open Prediction Page"):
            go_to_page("🔬 Prediction")
    else:
        input_raw = st.session_state.input_raw
        prediction_value = st.session_state.prediction
        probability = st.session_state.probability

        glucose = input_raw["Glucose"].values[0]
        bp = input_raw["BloodPressure"].values[0]
        skin = input_raw["SkinThickness"].values[0]
        insulin = input_raw["Insulin"].values[0]
        bmi = input_raw["BMI"].values[0]
        dpf = input_raw["DiabetesPedigreeFunction"].values[0]
        age = input_raw["Age"].values[0]

        st.markdown("### Prediction Result")
        st.markdown(f"**Patient Name:** {st.session_state.active_patient_name or 'Not provided'}")
        st.markdown(f"**Patient ID:** {st.session_state.active_patient_id or 'Not provided'}")

        if prediction_value == 1:
            result_text = "High Risk of Diabetes"
            st.error("High Risk of Diabetes — Please consult a doctor.")
            st.markdown('<span class="badge badge-red">HIGH RISK</span>', unsafe_allow_html=True)
        else:
            result_text = "Low Risk of Diabetes"
            st.success("Low Risk of Diabetes.")
            st.markdown('<span class="badge badge-green">LOW RISK</span>', unsafe_allow_html=True)

        if probability is not None:
            st.metric("Diabetes Risk Probability", f"{probability * 100:.2f}%")

        st.markdown("### Why this result?")
        for reason in get_risk_reasons(glucose, bmi, bp, insulin, prediction_value):
            st.markdown(f"- {reason}")

        st.markdown("### Patient Health Visualization")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Glucose", "BMI", "Blood Pressure", "Insulin", "Overall"
        ])

        with tab1:
            st.markdown("#### Glucose Risk Meter")
            fig, ax = plt.subplots(figsize=(5.0, 1.9))
            ax.set_xlim(70, 200)
            ax.set_ylim(0, 1)
            ax.hlines(0.5, 70, 99, linewidth=10, alpha=0.35, label="Normal")
            ax.hlines(0.5, 100, 125, linewidth=10, alpha=0.35, label="Prediabetes")
            ax.hlines(0.5, 126, 200, linewidth=10, alpha=0.35, label="High")
            ax.scatter([glucose], [0.5], s=220, marker="v", zorder=5)
            ax.text(glucose, 0.72, f"{glucose} mg/dL", ha="center", fontsize=10, fontweight="bold")
            ax.set_yticks([])
            ax.set_xlabel("Glucose range")
            ax.set_title("Doctor-Friendly Glucose Range Meter", fontsize=11)
            ax.spines[["left", "right", "top"]].set_visible(False)
            ax.legend(fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.55))
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)
            st.caption("The marker shows exactly where the patient's glucose falls: normal, prediabetic or high.")

        with tab2:
            st.markdown("#### BMI Level")
            fig, ax = plt.subplots(figsize=(4.8, 2.5))
            categories = ["Under", "Normal", "Over", "Obese"]
            limits = [18.5, 24.9, 29.9, 40]
            ax.plot(categories, limits, marker="o", linewidth=2, label="Category limit")
            patient_category = "Under" if bmi < 18.5 else "Normal" if bmi < 25 else "Over" if bmi < 30 else "Obese"
            ax.scatter([patient_category], [bmi], s=90, marker="D", label="Patient BMI")
            ax.set_ylabel("BMI")
            ax.set_title("BMI Category Position")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)
            st.caption("Shows the patient's BMI position in standard BMI categories.")

        with tab3:
            st.markdown("#### Blood Pressure Donut Chart")
            low_part = max(min(bp, 80), 0)
            high_part = max(120 - bp, 0)
            patient_part = max(120 - low_part - high_part, 1)
            values = [low_part, patient_part, high_part]
            labels = ["Lower Zone", "Patient Zone", "Remaining Zone"]
            fig, ax = plt.subplots(figsize=(3.8, 3.0))
            wedges, texts, autotexts = ax.pie(
                values,
                labels=labels,
                autopct="%1.0f%%",
                startangle=90,
                pctdistance=0.78,
                wedgeprops={"width": 0.38, "edgecolor": "white"},
                textprops={"fontsize": 8}
            )
            ax.text(0, 0, f"{bp}\nmm Hg", ha="center", va="center", fontsize=12, fontweight="bold")
            ax.set_title("Blood Pressure Status", fontsize=11)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)
            st.caption("A cleaner donut chart showing blood pressure as a compact professional visual.")

        with tab4:
            st.markdown("#### Insulin Level")
            fig, ax = plt.subplots(figsize=(4.8, 2.5))
            ax.fill_between([0, 1, 2], [0, insulin, 300], alpha=0.16)
            ax.plot(["Low", "Patient", "Upper"], [0, insulin, 300], marker="o", linewidth=2)
            ax.set_ylabel("μU/mL")
            ax.set_title("Insulin Range View")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)
            st.caption("Shows the insulin value in a simple range-style trend view.")

        with tab5:
            st.markdown("#### Overall Patient Profile")
            values = [glucose, bp, skin, insulin, bmi, dpf, age]
            labels = ["Glucose", "BP", "Skin", "Insulin", "BMI", "DPF", "Age"]

            fig, ax = plt.subplots(figsize=(5.3, 2.8))
            ax.plot(labels, values, marker="o", linewidth=2)
            ax.fill_between(labels, values, alpha=0.12)
            ax.set_title("Overall Health Profile", fontsize=11)
            ax.set_ylabel("Values")
            ax.grid(True, alpha=0.25)
            plt.xticks(rotation=20)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)
            st.caption("Overall graph is kept, but made smaller and cleaner.")

        st.markdown("### Medical Advice")

        show_medical_advice(glucose, bmi, bp, insulin, prediction_value)

        st.markdown("---")
        st.markdown("## Patient Report")

        patient_name_report = st.text_input(
            "Patient Name for Report",
            value=st.session_state.active_patient_name or "Unknown Patient"
        )
        patient_id_report = st.text_input(
            "Patient ID for Report",
            value=st.session_state.active_patient_id or "PT-001"
        )

        if st.button("Generate Report"):
            report_path = generate_patient_report(
                patient_name_report,
                patient_id_report,
                result_text,
                input_raw[['Pregnancies', 'Glucose', 'BloodPressure',
                           'SkinThickness', 'Insulin', 'BMI',
                           'DiabetesPedigreeFunction', 'Age']],
                st.session_state.patient_photo,
                include_photo_slot=(st.session_state.users_db.get(st.session_state.current_user, {}).get("role") == "Doctor"),
                report_for=st.session_state.users_db.get(st.session_state.current_user, {}).get("role", "Patient")
            )

            with open(report_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()

            st.success("Report generated successfully.")

            pdf_file_name = f"{patient_name_report}_report.pdf"
            whatsapp_message = quote(
                f"GLUCOTRACK Diabetes Prediction Report\n\n"
                f"Patient Name: {patient_name_report}\n"
                f"Patient ID: {patient_id_report}\n"
                f"Prediction Result: {result_text}\n"
                f"Glucose: {input_raw['Glucose'].values[0]} mg/dL\n"
                f"BMI: {input_raw['BMI'].values[0]} kg/m²\n"
                f"Blood Pressure: {input_raw['BloodPressure'].values[0]} mm Hg\n\n"
                f"PDF report has been generated. Download it and attach it in WhatsApp if direct file sharing is not supported."
            )
            whatsapp_url = f"https://wa.me/?text={whatsapp_message}"

            with st.expander("📤 Report Share", expanded=True):
                c_down, c_whats = st.columns(2)
                with c_down:
                    st.download_button(
                        label="⬇️ Download Report",
                        data=pdf_bytes,
                        file_name=pdf_file_name,
                        mime="application/pdf",
                        use_container_width=True
                    )
                with c_whats:
                    st.markdown(
                        f"""
                        <a href="{whatsapp_url}" target="_blank" style="text-decoration:none;">
                            <button style="
                                width:100%;
                                background:linear-gradient(135deg,#16A34A,#22C55E);
                                color:white;
                                border:none;
                                padding:12px 20px;
                                border-radius:12px;
                                cursor:pointer;
                                font-weight:800;
                                box-shadow:0 8px 20px rgba(34,197,94,0.25);">
                                🟢 Share on WhatsApp
                            </button>
                        </a>
                        """,
                        unsafe_allow_html=True
                    )
                st.caption("Direct PDF sharing depends on browser support. On desktop, download the PDF and attach it in WhatsApp.")
                build_whatsapp_file_share_button(pdf_bytes, pdf_file_name, f"GlucoTrack report for {patient_name_report}: {result_text}")

# ==============================
# ABOUT PAGE
# ==============================
elif menu == "ℹ️ About":
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.markdown("## About GlucoTrack")

    st.markdown("""
    <div class="card">
      <p>GlucoTrack is a diabetes risk prediction system powered by a supervised machine learning model.</p>
      <p>It uses patient medical values to predict whether the patient may have low or high diabetes risk.</p>
      <p><b>Important:</b> This tool is only for project and demo purposes. It should not replace medical advice from a qualified doctor.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Key Features")
    c1, c2, c3, c4 = st.columns(4)
    features = [
        ("Fast", "Prediction", "The system gives the diabetes risk result within a few seconds after entering patient values. This helps in quick screening and project demonstration."),
        ("8", "Main Inputs", "It uses 8 important medical inputs including glucose, BMI, blood pressure, insulin, age, pregnancies, skin thickness and diabetes pedigree function."),
        ("Secure", "Login", "Users can log in with their registered username and password. This keeps the project flow simple and professional."),
        ("PDF", "Report", "The app creates a professional patient report with result, input values, charts and health advice. The report can be downloaded and shared."),
    ]
    for col, (value, title, desc) in zip([c1, c2, c3, c4], features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
              <div class="feature-value">{value}</div>
              <div class="feature-title">{title}</div>
              <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ==============================
# SIDEBAR LOGOUT BUTTON
# ==============================
if st.session_state.get("logged_in", False):
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        try:
            st.query_params.clear()
        except Exception:
            pass
        st.session_state.patient_photo = None
        st.session_state.selected_menu = "🏠 Welcome"
        st.session_state.active_patient_name = ""
        st.session_state.active_patient_id = ""
        st.session_state.active_patient_age = None
        st.session_state.active_patient_gender = ""

        for key in ["prediction", "input_raw", "probability"]:
            if key in st.session_state:
                del st.session_state[key]

        st.rerun()
