import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import base64
import io
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from urllib.parse import quote
from datetime import datetime
import tempfile
import os
import random
import re

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
}

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Demo in-memory user store.
# For a real project, use a secure database and email/SMS OTP service.
if "users_db" not in st.session_state:
    st.session_state.users_db = {
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
    }

# ==============================
# THEME CSS
# ==============================
def apply_theme(theme):
    is_dark = theme == "Dark"

    bg = "#0B1220" if is_dark else "#F8FAFC"
    sidebar_bg = "#111827" if is_dark else "#E0F2FE"
    card_bg = "#172033" if is_dark else "#FFFFFF"
    text_main = "#F8FAFC" if is_dark else "#111827"
    text_sub = "#CBD5E1" if is_dark else "#475569"
    text_input = "#F8FAFC" if is_dark else "#111827"
    accent = "#0F766E"
    accent2 = "#0284C7"
    border = "#334155" if is_dark else "#BAE6FD"
    input_bg = "#1E293B" if is_dark else "#FFFFFF"
    hover_bg = "#0F766E" if is_dark else "#CFFAFE"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600;700&display=swap');

    html, body, .stApp {{
        background-color: {bg} !important;
        color: {text_main} !important;
        font-family: 'DM Sans', sans-serif;
    }}

    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
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
        border-radius: 10px !important;
        padding: 0.6rem 1.7rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.03em;
        transition: all 0.2s ease !important;
        box-shadow: 0 5px 18px rgba(15,118,110,0.28) !important;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 24px rgba(15,118,110,0.42) !important;
        filter: brightness(1.05) !important;
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
def go_to_page(page_name):
    st.session_state.selected_menu = page_name
    st.rerun()

def generate_otp():
    return str(random.randint(100000, 999999))


def clean_phone(value):
    return re.sub(r"\D", "", value or "")


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

def generate_patient_report(patient_name, patient_id, result_text, patient_data, patient_photo=None):
    safe_name = patient_name.replace(" ", "_") if patient_name else "patient"
    file_path = os.path.join(tempfile.gettempdir(), f"{safe_name}_diabetes_report.pdf")

    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Diabetes Prediction Patient Report")

    c.setFont("Helvetica", 10)
    c.drawString(50, height - 75, f"Generated On: {datetime.now().strftime('%d-%m-%Y %I:%M %p')}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 115, "Patient Details")

    c.setFont("Helvetica", 11)
    c.drawString(50, height - 140, f"Patient Name: {patient_name}")
    c.drawString(50, height - 160, f"Patient ID: {patient_id}")
    c.drawString(50, height - 180, f"Prediction Result: {result_text}")

    if patient_photo:
        try:
            photo_bytes = base64.b64decode(patient_photo)
            photo_img = ImageReader(io.BytesIO(photo_bytes))
            c.drawImage(photo_img, width - 160, height - 180, 90, 90)
        except Exception:
            pass

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 225, "Clinical Input Values")

    y = height - 250
    c.setFont("Helvetica", 10)

    for col in patient_data.columns:
        c.drawString(60, y, f"{col}: {patient_data[col].values[0]}")
        y -= 18

    c.showPage()

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Patient Data Visualization")

    chart1 = create_chart_image(
        "Glucose, BMI and Age",
        ["Glucose", "BMI", "Age"],
        [
            patient_data["Glucose"].values[0],
            patient_data["BMI"].values[0],
            patient_data["Age"].values[0]
        ]
    )

    chart2 = create_chart_image(
        "Blood Pressure, Insulin and Skin Thickness",
        ["BP", "Insulin", "Skin"],
        [
            patient_data["BloodPressure"].values[0],
            patient_data["Insulin"].values[0],
            patient_data["SkinThickness"].values[0]
        ]
    )

    c.drawImage(ImageReader(chart1), 50, height - 320, 480, 220)
    c.drawImage(ImageReader(chart2), 50, height - 570, 480, 220)

    c.showPage()

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Health Advice")

    glucose_value = patient_data["Glucose"].values[0]

    if "High" in result_text:
        advice_lines = [
            "1. Consult a doctor or diabetes specialist as soon as possible.",
            "2. Avoid excess sugar, sweet drinks and junk food.",
            "3. Walk for at least 30 minutes daily.",
            "4. Follow a balanced diet with vegetables and fiber-rich food.",
            "5. Monitor fasting and post-meal glucose regularly."
        ]
    elif 100 <= glucose_value < 126:
        advice_lines = [
            "1. The glucose value may be in the prediabetes range.",
            "2. Improve diet and reduce refined carbohydrates.",
            "3. Maintain a healthy body weight.",
            "4. Do regular physical exercise.",
            "5. Repeat the glucose test after medical consultation."
        ]
    else:
        advice_lines = [
            "1. Current risk appears low.",
            "2. Continue a healthy lifestyle.",
            "3. Maintain BMI in the normal range.",
            "4. Avoid excess sugar and processed food.",
            "5. Go for routine health checkups."
        ]

    y = height - 90
    c.setFont("Helvetica", 11)

    for line in advice_lines:
        c.drawString(60, y, line)
        y -= 25

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(
        50,
        50,
        "Note: This report is generated for project/demo purpose and is not a replacement for medical diagnosis."
    )

    c.save()
    return file_path

# ==============================
# SIDEBAR
# ==============================
st.sidebar.markdown("""
<div style='padding: 0.6rem 0 1rem 0;'>
  <span style='font-size:1.5rem; font-weight:800;'>🩺 GLUCOTRACK</span><br>
  <span style='font-size:0.75rem; opacity:0.55; letter-spacing:0.08em;'>
      SMART HEALTH DASHBOARD
  </span>
</div>
""", unsafe_allow_html=True)

menu_options = [
    "🏠 Welcome",
    "🔐 Login",
    "📝 Sign Up",
    "📋 Enroll Patient",
    "🔬 Prediction",
    "ℹ️ About"
]

default_index = 0
if st.session_state.selected_menu in menu_options:
    default_index = menu_options.index(st.session_state.selected_menu)

menu = st.sidebar.radio("Navigation", menu_options, index=default_index)
st.session_state.selected_menu = menu

st.sidebar.divider()

dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=(st.session_state.theme == "Dark"))
st.session_state.theme = "Dark" if dark_mode else "Light"
apply_theme(st.session_state.theme)

# ==============================
# WELCOME PAGE
# ==============================
if menu == "🏠 Welcome":
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Early Detection.<br>Better Health Decisions.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">GlucoTrack is a professional diabetes risk prediction system. It uses patient health values and a machine learning model to show a quick risk result and generate a patient report.</div>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Get Started →", use_container_width=False):
        st.session_state.selected_menu = "🔐 Login"
        st.rerun()

    st.markdown("### Key Features")
    c1, c2, c3, c4 = st.columns(4)
    features = [
        ("Fast", "Prediction", "The system gives the diabetes risk result in a few seconds after entering patient values."),
        ("8", "Main Inputs", "It uses 8 important medical inputs such as glucose, BMI, blood pressure, insulin and age."),
        ("OTP", "Login", "Users can log in using password or OTP with their registered email ID or phone number."),
        ("PDF", "Report", "The app can create a downloadable patient report with result, input values and advice."),
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

    st.markdown("---")
    st.info("Click Get Started to open the login page. Demo login: admin / 1234, or OTP using admin@gmail.com / 9999999999.")

# ==============================
# LOGIN PAGE WITH OTP
# ==============================
elif menu == "🔐 Login":
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.markdown("## Login")
    st.markdown("You can login using either your password or OTP.")

    tab_password, tab_otp = st.tabs(["Password Login", "OTP Login"])

    with tab_password:
        col_l, col_r = st.columns([1, 1])
        with col_l:
            login_user = st.text_input("Username", placeholder="Example: admin", key="password_login_user")
            login_password = st.text_input("Password", type="password", placeholder="Enter password", key="password_login_pass")

            if st.button("Login with Password →", use_container_width=True):
                db = st.session_state.users_db
                if login_user in db and str(db[login_user].get("password", "")) == login_password:
                    user_data = db[login_user]
                    st.session_state.logged_in = True
                    st.session_state.current_user = login_user
                    st.session_state.otp_sent = False
                    st.session_state.login_otp = None

                    if user_data.get("role") == "Doctor":
                        st.session_state.selected_menu = "📋 Enroll Patient"
                    else:
                        st.session_state.active_patient_name = user_data.get("full_name", "")
                        st.session_state.active_patient_id = user_data.get("patient_id", "")
                        st.session_state.active_patient_age = user_data.get("age")
                        st.session_state.active_patient_gender = user_data.get("gender", "")
                        st.session_state.selected_menu = "🔬 Prediction"

                    st.success(f"Welcome, {user_data.get('full_name', 'User')}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        with col_r:
            st.markdown("""
            <div class="card">
              <h4>Password Login</h4>
              <p>Use your registered username and password to access the app.</p>
              <p><b>Demo username:</b> admin<br><b>Demo password:</b> 1234</p>
            </div>
            """, unsafe_allow_html=True)

    with tab_otp:
        col_l, col_r = st.columns([1, 1])
        with col_l:
            login_identifier = st.text_input(
                "Email ID or Phone Number",
                value=st.session_state.login_identifier,
                placeholder="Example: admin@gmail.com or 9999999999",
                key="login_identifier_input"
            )

            if st.button("Send / Generate OTP", use_container_width=True):
                username_found, user_data = find_user_by_email_or_phone(login_identifier)

                if user_data:
                    st.session_state.login_identifier = login_identifier.strip()
                    st.session_state.login_username = username_found
                    st.session_state.login_otp = generate_otp()
                    st.session_state.otp_sent = True
                    st.rerun()
                else:
                    st.session_state.otp_sent = False
                    st.session_state.login_otp = None
                    st.session_state.login_username = None
                    st.error("No account found with this email ID or phone number. Please sign up first.")

            if st.session_state.otp_sent and st.session_state.login_otp:
                contact_type = "email ID" if "@" in st.session_state.login_identifier else "phone number"
                st.success(f"OTP generated for your registered {contact_type}.")
                st.info(f"Demo OTP for testing: {st.session_state.login_otp}")
                st.caption("For a real app, connect an email/SMS service. For this project demo, the OTP is shown here.")

                otp_input = st.text_input("Enter OTP", placeholder="Enter 6-digit OTP", key="otp_input")

                if st.button("Verify OTP and Login →", use_container_width=True):
                    if otp_input.strip() == str(st.session_state.login_otp):
                        db = st.session_state.users_db
                        user_data = db[st.session_state.login_username]

                        st.session_state.logged_in = True
                        st.session_state.current_user = st.session_state.login_username
                        st.session_state.otp_sent = False
                        st.session_state.login_otp = None

                        if user_data.get("role") == "Doctor":
                            st.session_state.selected_menu = "📋 Enroll Patient"
                        else:
                            st.session_state.active_patient_name = user_data.get("full_name", "")
                            st.session_state.active_patient_id = user_data.get("patient_id", "")
                            st.session_state.active_patient_age = user_data.get("age")
                            st.session_state.active_patient_gender = user_data.get("gender", "")
                            st.session_state.selected_menu = "🔬 Prediction"

                        st.success(f"Welcome, {user_data.get('full_name', 'User')}!")
                        st.rerun()
                    else:
                        st.error("Invalid OTP. Please enter the same OTP shown above.")
        with col_r:
            st.markdown("""
            <div class="card">
              <h4>OTP Login</h4>
              <p>You can login using your registered email ID or phone number.</p>
              <p>For project demo testing, the OTP is shown on the screen.</p>
              <p><b>Demo email:</b> admin@gmail.com<br><b>Demo phone:</b> 9999999999</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("New user? Open **Sign Up** from the sidebar.")

# ==============================
# SIGN UP PAGE
# ==============================
elif menu == "📝 Sign Up":
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.markdown("## Create Account")
    st.markdown("Create a Doctor or Patient account. After sign up, the correct page will open automatically.")
    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])

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
            su_patient_id = st.text_input("Patient ID", placeholder="Example: PT-20260001")
            su_age = st.number_input("Age", 1, 120, 25, key="su_age")
            su_gender = st.selectbox("Gender", ["Female", "Male", "Other"], key="su_gender")
            su_specialization = None
            su_license = None

        st.markdown("#### Contact Details for OTP Login")
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
                st.session_state.logged_in = True
                st.session_state.current_user = su_username

                if account_type == "Doctor":
                    st.session_state.selected_menu = "📋 Enroll Patient"
                else:
                    st.session_state.active_patient_name = su_fullname
                    st.session_state.active_patient_id = su_patient_id
                    st.session_state.active_patient_age = su_age
                    st.session_state.active_patient_gender = su_gender
                    st.session_state.selected_menu = "🔬 Prediction"

                st.success(f"{account_type} account created successfully.")
                st.balloons()
                st.rerun()


    with col_r:
        st.markdown("""
        <div class="card">
          <h4>Sign Up Rules</h4>
          <ul>
            <li>Both Doctor and Patient accounts can be created.</li>
            <li>Email ID, phone number and username must be unique.</li>
            <li>Password login and OTP login are both available.</li>
            <li>After sign up, the user will move directly to the correct page.</li>
            <li>Patients will move to the Prediction page.</li>
            <li>Doctors will move to the Patient Enrollment page.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Registered Users", expanded=False):
            for uname, udata in st.session_state.users_db.items():
                st.markdown(f"**{udata['full_name']}**  \n`{uname}` · {udata.get('role', 'User')}  \nEmail: `{udata.get('email', 'Not added')}`  \nPhone: `{udata.get('phone', 'Not added')}`")
                st.divider()

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

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("#### Patient Details")
                p_name = st.text_input("Patient Full Name", placeholder="Example: Ramesh Kumar")
                p_id = st.text_input("Patient ID", placeholder="Example: PT-20260001")
                p_age = st.number_input("Age", 1, 120, 35)
                p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                p_contact = st.text_input("Contact Number", placeholder="+91 XXXXX XXXXX")
                p_addr = st.text_area("Address", placeholder="City, State", height=70)
        
            with col2:
                st.markdown("#### Patient Photo")
                photo = st.file_uploader("Upload Photo (JPG / PNG)", type=["jpg", "jpeg", "png"])

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
                    b64 = base64.b64encode(buf.getvalue()).decode()
                    st.session_state.patient_photo = b64

                    st.markdown(f"""
                    <div style='text-align:center; margin-top:0.5rem;'>
                      <img src="data:image/png;base64,{b64}"
                           style="border-radius:14px; width:200px; height:200px;
                           object-fit:cover; border:3px solid #0F766E;
                           box-shadow:0 4px 20px rgba(15,118,110,0.35);" />
                      <div style='font-size:0.75rem; margin-top:0.5rem; opacity:0.6;'>Photo Preview</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No photo uploaded.")

                p_notes = st.text_area("Medical Notes", height=80, placeholder="Optional notes")
        
            if st.button("Enroll Patient and Open Prediction"):
                if not p_name or not p_id:
                    st.warning("Please enter Patient Name and Patient ID.")
                else:
                    st.session_state.active_patient_name = p_name
                    st.session_state.active_patient_id = p_id
                    st.session_state.active_patient_age = p_age
                    st.session_state.active_patient_gender = p_gender
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

        if not st.session_state.active_patient_name and current_user.get("role") == "Patient":
            st.session_state.active_patient_name = current_user.get("full_name", "")
            st.session_state.active_patient_id = current_user.get("patient_id", "")
            st.session_state.active_patient_age = current_user.get("age")
            st.session_state.active_patient_gender = current_user.get("gender", "")

        st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

        header_col, photo_col = st.columns([4, 1])

        with header_col:
            st.markdown("## Diabetes Risk Assessment")
            st.markdown("Enter the patient's health values and click **Run Prediction**.")

            patient_name = st.text_input(
                "Patient Name",
                value=st.session_state.active_patient_name or current_user.get("full_name", ""),
                placeholder="Enter patient name"
            )
            patient_id = st.text_input(
                "Patient ID",
                value=st.session_state.active_patient_id or "PT-001",
                placeholder="Enter patient ID"
            )

            st.session_state.active_patient_name = patient_name
            st.session_state.active_patient_id = patient_id

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
            preg = st.number_input("Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glucose (mg/dL)", 50, 200, 120)
            bp = st.number_input("Blood Pressure (mm Hg)", 30, 120, 70)
            skin = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    
        with col2:
            st.markdown("##### Body and Family Health Values")
            insulin = st.number_input("Insulin (μU/mL)", 0, 300, 100)
            bmi = st.number_input("BMI (kg/m²)", 10.0, 60.0, 25.0)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            default_age = int(st.session_state.active_patient_age) if st.session_state.active_patient_age else 30
            age = st.number_input("Age (years)", 1, 100, default_age)
    
        if st.button("Run Prediction"):
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

            input_encoded = pd.get_dummies(input_raw)
            input_df = input_encoded.reindex(columns=columns, fill_value=0)

            prediction = model.predict(input_df)

            try:
                probability = model.predict_proba(input_df)[0][1]
            except Exception:
                probability = None

            st.session_state.input_raw = input_raw
            st.session_state.prediction = prediction[0]
            st.session_state.probability = probability

        if "prediction" in st.session_state and "input_raw" in st.session_state:
            input_raw = st.session_state.input_raw
            prediction_value = st.session_state.prediction
            probability = st.session_state.probability

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
                    st.session_state.patient_photo
                )

                with open(report_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()

                st.download_button(
                    label="Download Report",
                    data=pdf_bytes,
                    file_name=f"{patient_name_report}_report.pdf",
                    mime="application/pdf"
                )

                whatsapp_message = quote(
                    f"Patient Report Generated\n"
                    f"Name: {patient_name_report}\n"
                    f"Patient ID: {patient_id_report}\n"
                    f"Result: {result_text}"
                )

                whatsapp_url = f"https://wa.me/?text={whatsapp_message}"

                st.markdown(
                    f"""
                    <a href="{whatsapp_url}" target="_blank">
                        <button style="
                            background-color:#25D366;
                            color:white;
                            border:none;
                            padding:10px 18px;
                            border-radius:8px;
                            cursor:pointer;
                            font-weight:600;
                            margin-top:10px;">
                            Share on WhatsApp
                        </button>
                    </a>
                    """,
                    unsafe_allow_html=True
                )

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

# ==============================
# SIDEBAR LOGOUT BUTTON
# ==============================
if st.session_state.get("logged_in", False):
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.session_state.patient_photo = None
        st.session_state.selected_menu = "🔐 Login"
        st.session_state.active_patient_name = ""
        st.session_state.active_patient_id = ""
        st.session_state.active_patient_age = None
        st.session_state.active_patient_gender = ""

        for key in ["prediction", "input_raw", "probability"]:
            if key in st.session_state:
                del st.session_state[key]

        st.rerun()
