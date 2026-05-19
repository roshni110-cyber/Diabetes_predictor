import streamlit as st
import pandas as pd
import pickle

# ==============================
# LOAD MODEL & COLUMNS
# ==============================
model = pickle.load(open("diabetes_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ==============================
# UI
# ==============================
st.title("🩺 Diabetes Prediction App")import streamlit as st
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
    "login_username": "",
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
            "specialization": "General Physician",
            "license": "DEMO-ADMIN"
        }
    }

# ==============================
# THEME CSS
# ==============================
def apply_theme(theme):
    is_dark = theme == "Dark"

    bg = "#0F1020" if is_dark else "#F5F3FF"
    sidebar_bg = "#17182E" if is_dark else "#FFFFFF"
    card_bg = "#1F213A" if is_dark else "#FFFFFF"
    text_main = "#F8FAFC" if is_dark else "#111827"
    text_sub = "#B8B9D6" if is_dark else "#6B7280"
    text_input = "#F8FAFC" if is_dark else "#111827"
    accent = "#7C3AED"
    accent2 = "#2563EB"
    border = "#343654" if is_dark else "#DDD6FE"
    input_bg = "#252847" if is_dark else "#FFFFFF"
    hover_bg = "#343654" if is_dark else "#EDE9FE"

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
        box-shadow: 0 5px 18px rgba(124,58,237,0.28) !important;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 24px rgba(124,58,237,0.42) !important;
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
        '<div class="hero-sub">GlucoTrack is a diabetes risk prediction system. It uses patient health values and a machine learning model to give a quick risk result.</div>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    tiles = [("Fast", "Prediction"), ("8", "Main Inputs"), ("OTP", "Login"), ("PDF", "Report")]
    for col, (val, lbl) in zip([c1, c2, c3, c4], tiles):
        with col:
            st.markdown(f"""
            <div class="metric-tile">
              <div class="val">{val}</div>
              <div class="lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("Please log in first. Demo username: admin. Click Send OTP on the login page.")

# ==============================
# LOGIN PAGE WITH OTP
# ==============================
elif menu == "🔐 Login":
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.markdown("## Login with OTP")
    st.markdown("Enter your username and verify the one-time password to access the system.")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        username = st.text_input(
            "Username",
            value=st.session_state.login_username,
            placeholder="Enter your username"
        )

        if st.button("Send OTP", use_container_width=True):
            db = st.session_state.users_db
            if username in db:
                st.session_state.login_username = username
                st.session_state.login_otp = generate_otp()
                st.session_state.otp_sent = True
                st.success("OTP generated successfully.")
                st.info(f"Demo OTP: {st.session_state.login_otp}")
            else:
                st.error("Username not found. Please sign up first.")

        if st.session_state.otp_sent:
            otp_input = st.text_input("Enter OTP", placeholder="Enter 6-digit OTP")

            if st.button("Verify OTP and Login →", use_container_width=True):
                if otp_input == st.session_state.login_otp:
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
                    st.error("Invalid OTP. Please check the demo OTP and try again.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("New user? Open **Sign Up** from the sidebar.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="card">
          <h4>How OTP Login Works</h4>
          <p>In this project version, the OTP is shown on the screen for demo testing.</p>
          <p>In a real application, the OTP should be sent through email or SMS.</p>
          <p><b>Demo username:</b> admin</p>
        </div>
        """, unsafe_allow_html=True)

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
        st.markdown('<div class="card">', unsafe_allow_html=True)

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

        st.markdown("#### Login Details")
        su_username = st.text_input("Choose Username", placeholder="Example: rose123", key="su_user")

        if st.button("Create Account", use_container_width=True):
            db = st.session_state.users_db

            if not su_fullname or not su_username:
                st.warning("Please fill all required fields.")
            elif len(su_username) < 3:
                st.error("Username must be at least 3 characters.")
            elif " " in su_username:
                st.error("Username cannot contain spaces.")
            elif su_username in db:
                st.error("Username already exists. Choose another username.")
            else:
                db[su_username] = {
                    "password": "",
                    "full_name": su_fullname,
                    "role": account_type,
                    "account_type": account_type,
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

        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="card">
          <h4>Sign Up Rules</h4>
          <ul>
            <li>Both Doctor and Patient accounts can be created.</li>
            <li>The username must be unique.</li>
            <li>After sign up, the user will move directly to the correct page.</li>
            <li>Patients will move to the Prediction page.</li>
            <li>Doctors will move to the Patient Enrollment page.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card"><h4>Registered Users</h4>', unsafe_allow_html=True)
        for uname, udata in st.session_state.users_db.items():
            st.markdown(f"**{udata['full_name']}**  \n`{uname}` · {udata.get('role', 'User')}")
            st.divider()
        st.markdown('</div>', unsafe_allow_html=True)

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
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Patient Details")
                p_name = st.text_input("Patient Full Name", placeholder="Example: Ramesh Kumar")
                p_id = st.text_input("Patient ID", placeholder="Example: PT-20260001")
                p_age = st.number_input("Age", 1, 120, 35)
                p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                p_contact = st.text_input("Contact Number", placeholder="+91 XXXXX XXXXX")
                p_addr = st.text_area("Address", placeholder="City, State", height=70)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
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
                           object-fit:cover; border:3px solid #7C3AED;
                           box-shadow:0 4px 20px rgba(124,58,237,0.35);" />
                      <div style='font-size:0.75rem; margin-top:0.5rem; opacity:0.6;'>Photo Preview</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No photo uploaded.")

                p_notes = st.text_area("Medical Notes", height=80, placeholder="Optional notes")
                st.markdown('</div>', unsafe_allow_html=True)

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
                       object-fit:cover; border:2px solid #7C3AED;" />
                </div>
                """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("##### Basic Health Values")
            preg = st.number_input("Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glucose (mg/dL)", 50, 200, 120)
            bp = st.number_input("Blood Pressure (mm Hg)", 30, 120, 70)
            skin = st.number_input("Skin Thickness (mm)", 0, 100, 20)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("##### Body and Family Health Values")
            insulin = st.number_input("Insulin (μU/mL)", 0, 300, 100)
            bmi = st.number_input("BMI (kg/m²)", 10.0, 60.0, 25.0)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            default_age = int(st.session_state.active_patient_age) if st.session_state.active_patient_age else 30
            age = st.number_input("Age (years)", 1, 100, default_age)
            st.markdown('</div>', unsafe_allow_html=True)

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
                st.markdown("#### Glucose Level Gauge")
                fig, ax = plt.subplots(figsize=(7, 2.8))
                ax.barh(["Glucose"], [glucose])
                ax.axvline(100, linestyle="--", label="Normal Limit")
                ax.axvline(126, linestyle="--", label="Diabetes Check Level")
                ax.set_xlim(50, 200)
                ax.set_xlabel("mg/dL")
                ax.set_title("Patient Glucose Level")
                ax.legend()
                st.pyplot(fig)
                st.caption("This chart compares the patient's glucose value with common clinical reference points.")

            with tab2:
                st.markdown("#### BMI Category Chart")
                categories = ["Underweight", "Normal", "Overweight", "Obese"]
                bmi_ranges = [18.5, 6.4, 4.9, 10]
                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.bar(categories, bmi_ranges)
                ax.scatter(["Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"], [bmi], s=120, marker="D", label="Patient BMI")
                ax.axhline(bmi, linestyle="--")
                ax.set_ylabel("BMI Value")
                ax.set_title("BMI Category Comparison")
                ax.legend()
                st.pyplot(fig)
                st.caption("This chart shows where the patient's BMI falls compared with common BMI categories.")

            with tab3:
                st.markdown("#### Blood Pressure Status")
                labels = ["Low", "Normal", "High"]
                values = [60, 80, 100]
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.pie(values, labels=labels, autopct="%1.0f%%", startangle=90)
                ax.set_title(f"Patient Blood Pressure: {bp} mm Hg")
                st.pyplot(fig)
                st.caption("This pie chart gives a simple visual reference for low, normal and high blood pressure zones.")

            with tab4:
                st.markdown("#### Insulin Level Trend View")
                points = ["Start", "Patient", "Reference"]
                insulin_values = [0, insulin, 300]
                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.plot(points, insulin_values, marker="o", linewidth=2)
                ax.fill_between(points, insulin_values, alpha=0.15)
                ax.set_ylabel("μU/mL")
                ax.set_title("Insulin Level View")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                st.caption("This chart gives a trend-style view of the patient's insulin level.")

            with tab5:
                st.markdown("#### Overall Patient Profile")
                values = [glucose, bp, skin, insulin, bmi, dpf, age]
                labels = ["Glucose", "BP", "Skin", "Insulin", "BMI", "DPF", "Age"]

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(labels, values, marker="o", linewidth=2)
                ax.fill_between(labels, values, alpha=0.15)
                ax.set_title("Overall Patient Health Profile")
                ax.set_ylabel("Values")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                st.caption("This overall chart shows all selected patient values in one combined profile.")

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

st.write("Enter patient details below:")

# Inputs
preg = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 50, 200, 120)
bp = st.number_input("Blood Pressure", 30, 120, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 300, 100)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 100, 30)

# ==============================
# PREDICTION
# ==============================
if st.button("Predict"):

    # --------------------------
    # Create input dataframe
    # --------------------------
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

    # --------------------------
    # Feature Engineering
    # --------------------------
    input_raw['Glucose_BMI'] = input_raw['Glucose'] * input_raw['BMI']
    input_raw['Insulin_Glucose'] = input_raw['Insulin'] * input_raw['Glucose']
    input_raw['Age_BMI'] = input_raw['Age'] * input_raw['BMI']
    input_raw['BMI_Squared'] = input_raw['BMI'] ** 2

    # --------------------------
    # Encoding
    # --------------------------
    input_encoded = pd.get_dummies(input_raw)

    # --------------------------
    # Match training columns
    # --------------------------
    input_df = input_encoded.reindex(columns=columns, fill_value=0)

    # --------------------------
    # Prediction
    # --------------------------
    prediction = model.predict(input_df)

    # --------------------------
    # Output
    # --------------------------
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")
        
