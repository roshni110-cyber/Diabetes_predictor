import streamlit as st
import pandas as pd
import pickle
import os
import io
import base64
import tempfile
from datetime import datetime
from urllib.parse import quote

import matplotlib.pyplot as plt
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="GlucoTrack — Diabetes Risk App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==============================
# SESSION STATE
# ==============================
def init_state():
    defaults = {
        "logged_in": False,
        "current_user": None,
        "theme": "Light",
        "selected_menu": "Welcome",
        "patient_photo": None,
        "selected_patient_id": None,
        "prediction_done": False,
        "latest_input": None,
        "latest_result": None,
        "latest_probability": None,
        "patients_db": {},
        "users_db": {
            "admin": {
                "password": "1234",
                "full_name": "Administrator",
                "role": "Doctor",
                "email": "admin@gmail.com",
                "phone": "9999999999"
            }
        }
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("diabetes_model.pkl", "rb"))
        columns = pickle.load(open("columns.pkl", "rb"))
        return model, columns
    except Exception:
        return None, None


model, columns = load_model()


# ==============================
# THEME CSS
# ==============================
def apply_theme(theme):
    dark = theme == "Dark"

    bg = "#EAF7FF" if not dark else "#071A2D"
    card = "#FFFFFF" if not dark else "#102A43"
    sidebar = "#DFF3FF" if not dark else "#0B2239"
    text = "#0F172A" if not dark else "#F8FAFC"
    subtext = "#475569" if not dark else "#CBD5E1"
    border = "#BFE7FF" if not dark else "#1E3A5F"
    input_bg = "#FFFFFF" if not dark else "#0F2A44"
    accent = "#0284C7" if not dark else "#38BDF8"
    accent2 = "#0EA5E9" if not dark else "#2563EB"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, .stApp {{
        background: {bg} !important;
        color: {text} !important;
        font-family: 'Inter', sans-serif !important;
    }}

    .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1150px !important;
    }}

    section[data-testid="stSidebar"] {{
        background: {sidebar} !important;
        border-right: 1px solid {border} !important;
    }}

    section[data-testid="stSidebar"] * {{
        color: {text} !important;
    }}

    h1, h2, h3, h4, h5, h6, p, label, span, div {{
        color: {text} !important;
        font-family: 'Inter', sans-serif !important;
    }}

    p {{
        color: {subtext} !important;
        line-height: 1.6 !important;
    }}

    .main-card {{
        background: {card};
        border: 1px solid {border};
        border-radius: 22px;
        padding: 1.4rem;
        box-shadow: 0 12px 35px rgba(2,132,199,0.12);
        margin-bottom: 1rem;
    }}

    .mini-card {{
        background: {card};
        border: 1px solid {border};
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 8px 25px rgba(2,132,199,0.10);
        height: 100%;
    }}

    .hero-box {{
        text-align: center;
        padding: 2rem 1rem 1rem 1rem;
    }}

    .hero-title {{
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0.6rem;
    }}

    .hero-sub {{
        font-size: 1.05rem;
        color: {subtext} !important;
        max-width: 620px;
        margin: auto;
    }}

    .gluco-logo {{
        font-size: 1.7rem;
        font-weight: 800;
        letter-spacing: 0.03em;
        margin-bottom: 0.2rem;
    }}

    .accent-line {{
        height: 4px;
        width: 72px;
        background: linear-gradient(135deg, {accent}, {accent2});
        border-radius: 20px;
        margin-bottom: 1.3rem;
    }}

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextArea"] textarea {{
        background: {input_bg} !important;
        color: {text} !important;
        border: 1px solid {border} !important;
        border-radius: 12px !important;
        min-height: 40px !important;
        box-shadow: none !important;
        font-size: 0.95rem !important;
    }}

    div[data-testid="stTextInput"] button {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}

    div[data-baseweb="select"] > div {{
        background: {input_bg} !important;
        color: {text} !important;
        border: 1px solid {border} !important;
        border-radius: 12px !important;
        min-height: 40px !important;
    }}

    .stButton > button {{
        width: auto !important;
        min-width: 110px !important;
        height: 38px !important;
        padding: 0.35rem 1.1rem !important;
        border-radius: 12px !important;
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, {accent}, {accent2}) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 8px 22px rgba(2,132,199,0.25) !important;
        cursor: pointer !important;
        transition: 0.2s ease !important;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 12px 26px rgba(2,132,199,0.34) !important;
    }}

    .stDownloadButton > button {{
        width: auto !important;
        min-width: 130px !important;
        height: 38px !important;
        border-radius: 12px !important;
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #0891B2, #0284C7) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 8px 22px rgba(2,132,199,0.22) !important;
    }}

    .login-wrapper {{
        max-width: 420px;
        margin: 2rem auto;
        text-align: left;
    }}

    .login-title {{
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 1.2rem;
    }}

    .center-btn {{
        text-align: center;
        margin-top: 0.8rem;
    }}

    .metric-box {{
        text-align: center;
        padding: 1rem;
        border-radius: 18px;
        background: {card};
        border: 1px solid {border};
        box-shadow: 0 8px 22px rgba(2,132,199,0.10);
    }}

    .metric-value {{
        font-size: 1.6rem;
        font-weight: 800;
        color: {accent} !important;
    }}

    .metric-label {{
        font-size: 0.82rem;
        color: {subtext} !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .risk-red {{
        background: rgba(239,68,68,0.12);
        color: #EF4444 !important;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        font-weight: 800;
        display: inline-block;
    }}

    .risk-green {{
        background: rgba(34,197,94,0.14);
        color: #22C55E !important;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        font-weight: 800;
        display: inline-block;
    }}

    .whatsapp-btn {{
        display: inline-block;
        background: #25D366;
        color: white !important;
        padding: 9px 16px;
        border-radius: 12px;
        text-decoration: none;
        font-weight: 800;
        font-size: 0.92rem;
        margin-top: 0.5rem;
        box-shadow: 0 8px 22px rgba(37,211,102,0.25);
    }}

    .whatsapp-btn:hover {{
        filter: brightness(0.95);
    }}

    div[data-testid="stToolbar"] {{
        visibility: visible !important;
        opacity: 1 !important;
    }}

    div[data-testid="stToolbar"] button,
    button[kind="icon"] {{
        color: {text} !important;
        background: {card} !important;
        border: 1px solid {border} !important;
    }}

    @media (max-width: 768px) {{
        .block-container {{
            padding: 1rem !important;
        }}

        .hero-title {{
            font-size: 2.1rem !important;
        }}

        .login-wrapper {{
            max-width: 95% !important;
        }}

        .stButton > button {{
            min-width: 95px !important;
            height: 36px !important;
            font-size: 0.85rem !important;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)


apply_theme(st.session_state.theme)


# ==============================
# HELPER FUNCTIONS
# ==============================
def get_current_user():
    if not st.session_state.current_user:
        return {}
    return st.session_state.users_db.get(st.session_state.current_user, {})


def current_role():
    return get_current_user().get("role", "")


def current_doctor_key():
    return st.session_state.current_user or "unknown_doctor"


def get_doctor_patients():
    doctor = current_doctor_key()
    if doctor not in st.session_state.patients_db:
        st.session_state.patients_db[doctor] = {}
    return st.session_state.patients_db[doctor]


def create_chart_image(title, labels, values):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    buffer.seek(0)
    plt.close(fig)
    return buffer


def generate_patient_report(patient_name, patient_id, result_text, patient_data, patient_photo=None, doctor_report=False):
    safe_name = patient_name.replace(" ", "_")
    file_path = os.path.join(tempfile.gettempdir(), f"{safe_name}_diabetes_report.pdf")

    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4

    primary = colors.HexColor("#0284C7")
    dark_text = colors.HexColor("#0F172A")
    light_bg = colors.HexColor("#EAF7FF")
    red = colors.HexColor("#EF4444")
    green = colors.HexColor("#16A34A")

    c.setFillColor(primary)
    c.rect(0, height - 95, width, 95, fill=1, stroke=0)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(45, height - 45, "GlucoTrack Diabetes Report")

    c.setFont("Helvetica", 10)
    c.drawString(45, height - 65, f"Generated on: {datetime.now().strftime('%d-%m-%Y %I:%M %p')}")

    c.setFillColor(light_bg)
    c.roundRect(40, height - 245, width - 80, 120, 14, fill=1, stroke=0)

    c.setFillColor(dark_text)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(60, height - 150, "Patient Summary")

    c.setFont("Helvetica", 11)
    c.drawString(60, height - 175, f"Patient Name: {patient_name}")

    if doctor_report:
        c.drawString(60, height - 195, f"Patient ID: {patient_id}")
    else:
        c.drawString(60, height - 195, "Patient Type: Self Check")

    c.drawString(60, height - 215, f"Prediction Result: {result_text}")

    badge_color = red if "High" in result_text else green
    c.setFillColor(badge_color)
    c.roundRect(width - 210, height - 205, 130, 34, 14, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(width - 145, height - 193, "HIGH RISK" if "High" in result_text else "LOW RISK")

    if doctor_report and patient_photo:
        try:
            img_bytes = base64.b64decode(patient_photo)
            img_reader = ImageReader(io.BytesIO(img_bytes))
            c.drawImage(img_reader, width - 150, height - 165, 75, 75, mask="auto")
        except Exception:
            pass

    c.setFillColor(dark_text)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(45, height - 285, "Clinical Input Values")

    y = height - 315
    c.setFont("Helvetica", 10)

    for col in patient_data.columns:
        c.drawString(60, y, f"{col}: {patient_data[col].values[0]}")
        y -= 18

    c.showPage()

    c.setFillColor(primary)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(45, height - 55, "Health Visualization")

    chart1 = create_chart_image(
        "Key Risk Factors",
        ["Glucose", "BMI", "BP", "Insulin"],
        [
            patient_data["Glucose"].values[0],
            patient_data["BMI"].values[0],
            patient_data["BloodPressure"].values[0],
            patient_data["Insulin"].values[0]
        ]
    )

    c.drawImage(ImageReader(chart1), 50, height - 365, 500, 260)

    c.showPage()

    c.setFillColor(primary)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(45, height - 55, "Medical Advice")

    g = patient_data["Glucose"].values[0]
    bmi = patient_data["BMI"].values[0]
    insulin = patient_data["Insulin"].values[0]
    bp = patient_data["BloodPressure"].values[0]

    advice = get_medical_advice(g, bmi, insulin, bp, 1 if "High" in result_text else 0)

    y = height - 95
    c.setFillColor(dark_text)
    c.setFont("Helvetica", 11)

    for line in advice:
        c.drawString(55, y, f"- {line}")
        y -= 24

    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.grey)
    c.drawString(
        45,
        45,
        "Note: This report is for project/demo purpose and is not a replacement for medical diagnosis."
    )

    c.save()
    return file_path


def get_risk_reason(glucose, bmi, insulin, bp, prediction):
    reasons = []

    if glucose >= 126:
        reasons.append("high glucose level")
    elif 100 <= glucose < 126:
        reasons.append("prediabetic glucose range")

    if bmi >= 30:
        reasons.append("obesity range BMI")
    elif bmi >= 25:
        reasons.append("overweight BMI")

    if insulin > 200:
        reasons.append("high insulin level")

    if bp >= 90:
        reasons.append("high blood pressure")

    if prediction == 1 and not reasons:
        reasons.append("combined medical input pattern")

    if not reasons:
        return "No major high-risk factor is visible from the entered values."

    return "Risk is mainly due to " + ", ".join(reasons) + "."


def get_medical_advice(glucose, bmi, insulin, bp, prediction):
    advice = []

    if prediction == 1:
        advice.append("Consult a doctor or diabetologist for proper medical evaluation.")

    if glucose >= 126:
        advice.append("Glucose is high. Avoid sugary drinks, sweets, and refined carbohydrates.")
        advice.append("Monitor fasting and post-meal glucose regularly.")
    elif 100 <= glucose < 126:
        advice.append("Glucose is in the prediabetic range. Improve diet and increase physical activity.")
        advice.append("Repeat glucose testing after medical consultation.")
    else:
        advice.append("Glucose level is currently not in a high-risk range. Maintain a healthy routine.")

    if bmi >= 30:
        advice.append("BMI is in the obesity range. Focus on gradual weight reduction.")
    elif bmi >= 25:
        advice.append("BMI is slightly high. Maintain calorie control and regular exercise.")

    if insulin > 200:
        advice.append("Insulin level is high. Medical review is advised.")

    if bp >= 90:
        advice.append("Blood pressure is high. Reduce salt intake and check BP regularly.")

    advice.append("Walk at least 30 minutes daily if medically allowed.")
    advice.append("This app supports screening only; final diagnosis must be done by a doctor.")

    return advice


# ==============================
# SIDEBAR / NAVIGATION
# ==============================
def render_sidebar():
    if not st.session_state.logged_in:
        return

    st.sidebar.markdown("""
    <div style='padding: 1rem 0;'>
        <div style='font-size:1.45rem;font-weight:800;'>🩺 GLUCOTRACK</div>
        <div style='font-size:0.75rem;opacity:0.65;letter-spacing:0.08em;'>SMART HEALTH DASHBOARD</div>
    </div>
    """, unsafe_allow_html=True)

    role = current_role()

    if role == "Doctor":
        menu_options = [
            "Doctor Home",
            "Enroll Patient",
            "Patient Details",
            "Prediction",
            "Visualization",
            "About"
        ]
    else:
        menu_options = [
            "Prediction",
            "Visualization",
            "About"
        ]

    if st.session_state.selected_menu not in menu_options:
        st.session_state.selected_menu = menu_options[0]

    st.session_state.selected_menu = st.sidebar.radio(
        "Navigation",
        menu_options,
        index=menu_options.index(st.session_state.selected_menu)
    )

    st.sidebar.divider()

    dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.theme == "Dark")
    st.session_state.theme = "Dark" if dark_mode else "Light"

    st.sidebar.divider()

    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.session_state.selected_menu = "Welcome"
        st.session_state.selected_patient_id = None
        st.session_state.prediction_done = False
        st.rerun()


render_sidebar()
apply_theme(st.session_state.theme)


# ==============================
# WELCOME PAGE
# ==============================
if not st.session_state.logged_in and st.session_state.selected_menu == "Welcome":
    st.markdown("""
    <div class='hero-box'>
        <div class='gluco-logo'>🩺 GLUCOTRACK</div>
        <div class='hero-title'>Smart Diabetes Risk Assessment</div>
        <p class='hero-sub'>
            A simple and professional health dashboard for patient enrollment,
            diabetes risk prediction, visualization, and report generation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("""
        <div class='main-card' style='text-align:center;'>
            <div style='font-size:5rem;'>🩸</div>
            <h3>Clinical Prediction Made Easier</h3>
            <p>Designed for doctors and patients to check diabetes risk using health values.</p>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Login", use_container_width=True):
                st.session_state.selected_menu = "Login"
                st.rerun()
        with col_b:
            if st.button("Create Account", use_container_width=True):
                st.session_state.selected_menu = "Sign Up"
                st.rerun()


# ==============================
# LOGIN PAGE
# ==============================
elif not st.session_state.logged_in and st.session_state.selected_menu == "Login":
    st.markdown("<div class='login-wrapper'>", unsafe_allow_html=True)
    st.markdown("<div class='login-title'>Login</div>", unsafe_allow_html=True)

    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")

    st.markdown("<div class='center-btn'>", unsafe_allow_html=True)
    login_clicked = st.button("Login")
    st.markdown("</div>", unsafe_allow_html=True)

    if login_clicked:
        db = st.session_state.users_db
        if username in db and db[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.current_user = username

            if db[username]["role"] == "Doctor":
                st.session_state.selected_menu = "Doctor Home"
            else:
                st.session_state.selected_menu = "Prediction"

            st.success("Login successful.")
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.markdown("<div style='text-align:center;margin-top:1rem;'>", unsafe_allow_html=True)
    if st.button("Create New Account"):
        st.session_state.selected_menu = "Sign Up"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================
# SIGN UP PAGE
# ==============================
elif not st.session_state.logged_in and st.session_state.selected_menu == "Sign Up":
    st.markdown("<div class='login-wrapper'>", unsafe_allow_html=True)
    st.markdown("<div class='login-title'>Create Account</div>", unsafe_allow_html=True)

    account_type = st.radio("Account Type", ["Patient", "Doctor"], horizontal=True)
    full_name = st.text_input("Full Name", placeholder="Enter full name")
    email = st.text_input("Email", placeholder="Enter email")
    phone = st.text_input("Phone Number", placeholder="Enter phone number")
    username = st.text_input("Username", placeholder="Choose username")
    password = st.text_input("Password", type="password", placeholder="Choose password")
    confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm password")

    age = None
    gender = None

    if account_type == "Patient":
        age = st.number_input("Age", 1, 120, 25)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])

    st.markdown("<div class='center-btn'>", unsafe_allow_html=True)
    signup_clicked = st.button("Create Account")
    st.markdown("</div>", unsafe_allow_html=True)

    if signup_clicked:
        db = st.session_state.users_db

        if not full_name or not username or not password:
            st.warning("Please fill all required fields.")
        elif username in db:
            st.error("Username already exists.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        elif len(password) < 4:
            st.error("Password must be at least 4 characters.")
        else:
            db[username] = {
                "password": password,
                "full_name": full_name,
                "role": account_type,
                "email": email,
                "phone": phone,
                "age": age,
                "gender": gender
            }

            st.session_state.users_db = db
            st.session_state.logged_in = True
            st.session_state.current_user = username

            if account_type == "Doctor":
                st.session_state.selected_menu = "Doctor Home"
            else:
                st.session_state.selected_menu = "Prediction"

            st.success("Account created successfully.")
            st.rerun()

    st.markdown("<div style='text-align:center;margin-top:1rem;'>", unsafe_allow_html=True)
    if st.button("Back to Login"):
        st.session_state.selected_menu = "Login"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================
# DOCTOR HOME
# ==============================
elif st.session_state.logged_in and st.session_state.selected_menu == "Doctor Home":
    st.markdown("<div class='accent-line'></div>", unsafe_allow_html=True)
    st.title("Welcome Doctor")

    doctor_name = get_current_user().get("full_name", "Doctor")
    st.markdown(f"""
    <div class='main-card'>
        <h3>Hello, Dr. {doctor_name}</h3>
        <p>
        Use this dashboard to enroll patients, review saved patient records,
        run diabetes risk prediction, view health visualization, and generate reports.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    patients = get_doctor_patients()

    with c1:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-value'>{len(patients)}</div>
            <div class='metric-label'>Enrolled Patients</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        checked = sum(1 for p in patients.values() if p.get("last_checked"))
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-value'>{checked}</div>
            <div class='metric-label'>Checked Patients</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class='metric-box'>
            <div class='metric-value'>ML</div>
            <div class='metric-label'>Prediction System</div>
        </div>
        """, unsafe_allow_html=True)


# ==============================
# ENROLL PATIENT
# ==============================
elif st.session_state.logged_in and st.session_state.selected_menu == "Enroll Patient":
    if current_role() != "Doctor":
        st.warning("Only doctors can access this page.")
    else:
        st.markdown("<div class='accent-line'></div>", unsafe_allow_html=True)
        st.title("Enroll Patient")
        st.write("Add a new patient record. Patient ID must be unique for this doctor.")

        patients = get_doctor_patients()

        col1, col2 = st.columns(2)

        with col1:
            patient_id = st.text_input("Patient ID", placeholder="Example: PT-001")
            patient_name = st.text_input("Patient Name", placeholder="Enter patient name")
            age = st.number_input("Age", 1, 120, 30)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            contact = st.text_input("Contact Number", placeholder="Enter contact number")

        with col2:
            address = st.text_area("Address", placeholder="Enter address")
            notes = st.text_area("Notes", placeholder="Optional notes")
            photo = st.file_uploader("Patient Photo", type=["png", "jpg", "jpeg"])

            photo_b64 = None
            if photo:
                img = Image.open(photo)
                img = img.resize((220, 220))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                photo_b64 = base64.b64encode(buf.getvalue()).decode()
                st.image(img, width=140)

        if st.button("Enroll Patient"):
            if not patient_id or not patient_name:
                st.warning("Patient ID and Patient Name are required.")
            elif patient_id in patients:
                st.error("This Patient ID already exists for this doctor.")
            else:
                patients[patient_id] = {
                    "patient_id": patient_id,
                    "name": patient_name,
                    "age": age,
                    "gender": gender,
                    "contact": contact,
                    "address": address,
                    "notes": notes,
                    "photo": photo_b64,
                    "clinical_data": {},
                    "last_checked": "",
                    "result": ""
                }

                st.session_state.patients_db[current_doctor_key()] = patients
                st.session_state.selected_patient_id = patient_id
                st.success("Patient enrolled successfully.")
                st.session_state.selected_menu = "Prediction"
                st.rerun()


# ==============================
# PATIENT DETAILS
# ==============================
elif st.session_state.logged_in and st.session_state.selected_menu == "Patient Details":
    if current_role() != "Doctor":
        st.warning("Only doctors can access this page.")
    else:
        st.markdown("<div class='accent-line'></div>", unsafe_allow_html=True)
        st.title("Patient Details")
        st.write("These are the patients enrolled by the logged-in doctor only.")

        patients = get_doctor_patients()

        search = st.text_input("Search by Patient ID or Name", placeholder="Example: PT-001 or Ramesh")

        rows = []
        for pid, p in patients.items():
            if search.lower() in pid.lower() or search.lower() in p.get("name", "").lower():
                data = p.get("clinical_data", {})
                rows.append({
                    "Patient ID": pid,
                    "Name": p.get("name", ""),
                    "Age": p.get("age", ""),
                    "Gender": p.get("gender", ""),
                    "Contact": p.get("contact", ""),
                    "Last Glucose": data.get("Glucose", ""),
                    "Last BMI": data.get("BMI", ""),
                    "Last Checked": p.get("last_checked", ""),
                    "Result": p.get("result", ""),
                    "Notes": p.get("notes", "")
                })

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            patient_ids = [row["Patient ID"] for row in rows]
            selected_id = st.selectbox("Select Patient ID to open", patient_ids)

            c1, c2 = st.columns([1, 1])

            with c1:
                if st.button("Open in Prediction"):
                    st.session_state.selected_patient_id = selected_id
                    p = patients[selected_id]

                    if p.get("clinical_data"):
                        st.session_state.latest_input = pd.DataFrame([p["clinical_data"]])
                        st.session_state.prediction_done = True

                    st.session_state.selected_menu = "Prediction"
                    st.rerun()

            with c2:
                if st.button("Delete Patient"):
                    del patients[selected_id]
                    st.session_state.patients_db[current_doctor_key()] = patients

                    if st.session_state.selected_patient_id == selected_id:
                        st.session_state.selected_patient_id = None

                    st.success("Patient deleted successfully.")
                    st.rerun()
        else:
            st.info("No patient record found.")


# ==============================
# PREDICTION PAGE
# ==============================
elif st.session_state.logged_in and st.session_state.selected_menu == "Prediction":
    st.markdown("<div class='accent-line'></div>", unsafe_allow_html=True)
    st.title("Diabetes Risk Assessment")

    role = current_role()
    user = get_current_user()
    patients = get_doctor_patients() if role == "Doctor" else {}

    selected_patient = None

    if role == "Doctor":
        if st.session_state.selected_patient_id and st.session_state.selected_patient_id in patients:
            selected_patient = patients[st.session_state.selected_patient_id]
            st.info(f"Selected Patient: {selected_patient['name']} ({st.session_state.selected_patient_id})")
        else:
            st.warning("Please enroll or open a patient first.")
    else:
        selected_patient = {
            "name": user.get("full_name", "Patient"),
            "age": user.get("age", 25),
            "gender": user.get("gender", "Female"),
            "patient_id": "SELF"
        }
        st.info(f"Patient: {selected_patient['name']}")

    saved_data = selected_patient.get("clinical_data", {}) if selected_patient else {}

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.subheader("Basic Medical Values")
        preg = st.number_input("Pregnancies", 0, 20, int(saved_data.get("Pregnancies", 1)))
        glucose = st.number_input("Glucose (mg/dL)", 50, 250, int(saved_data.get("Glucose", 120)))
        bp = st.number_input("Blood Pressure (mm Hg)", 30, 140, int(saved_data.get("BloodPressure", 70)))
        skin = st.number_input("Skin Thickness (mm)", 0, 100, int(saved_data.get("SkinThickness", 20)))
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.subheader("Body and Family Health")
        insulin = st.number_input("Insulin (μU/mL)", 0, 400, int(saved_data.get("Insulin", 100)))
        bmi = st.number_input("BMI", 10.0, 70.0, float(saved_data.get("BMI", 25.0)))
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, float(saved_data.get("DiabetesPedigreeFunction", 0.5)))
        age = st.number_input("Age", 1, 120, int(saved_data.get("Age", selected_patient.get("age", 30) if selected_patient else 30)))
        st.markdown("</div>", unsafe_allow_html=True)

    input_raw = pd.DataFrame({
        "Pregnancies": [preg],
        "Glucose": [glucose],
        "BloodPressure": [bp],
        "SkinThickness": [skin],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [dpf],
        "Age": [age]
    })

    c1, c2 = st.columns([1, 1])

    with c1:
        if st.button("Save Patient Data"):
            clinical_data = input_raw.iloc[0].to_dict()

            if role == "Doctor" and selected_patient:
                pid = st.session_state.selected_patient_id
                patients[pid]["clinical_data"] = clinical_data
                patients[pid]["last_checked"] = datetime.now().strftime("%d-%m-%Y %I:%M %p")
                st.session_state.patients_db[current_doctor_key()] = patients
            else:
                st.session_state.users_db[st.session_state.current_user]["clinical_data"] = clinical_data
                st.session_state.users_db[st.session_state.current_user]["last_checked"] = datetime.now().strftime("%d-%m-%Y %I:%M %p")

            st.success("Patient data saved successfully.")

    with c2:
        run_prediction = st.button("Run Prediction")

    if run_prediction:
        if model is None or columns is None:
            st.error("Model files not found. Please keep diabetes_model.pkl and columns.pkl in the same folder.")
        else:
            model_input = input_raw.copy()
            model_input["Glucose_BMI"] = model_input["Glucose"] * model_input["BMI"]
            model_input["Insulin_Glucose"] = model_input["Insulin"] * model_input["Glucose"]
            model_input["Age_BMI"] = model_input["Age"] * model_input["BMI"]
            model_input["BMI_Squared"] = model_input["BMI"] ** 2

            encoded = pd.get_dummies(model_input)
            input_df = encoded.reindex(columns=columns, fill_value=0)

            prediction = model.predict(input_df)[0]

            try:
                probability = model.predict_proba(input_df)[0][1]
            except Exception:
                probability = None

            result_text = "High Risk of Diabetes" if prediction == 1 else "Low Risk of Diabetes"

            st.session_state.latest_input = input_raw
            st.session_state.latest_result = result_text
            st.session_state.latest_probability = probability
            st.session_state.prediction_done = True

            clinical_data = input_raw.iloc[0].to_dict()

            if role == "Doctor" and selected_patient:
                pid = st.session_state.selected_patient_id
                patients[pid]["clinical_data"] = clinical_data
                patients[pid]["last_checked"] = datetime.now().strftime("%d-%m-%Y %I:%M %p")
                patients[pid]["result"] = result_text
                st.session_state.patients_db[current_doctor_key()] = patients
            else:
                st.session_state.users_db[st.session_state.current_user]["clinical_data"] = clinical_data
                st.session_state.users_db[st.session_state.current_user]["last_checked"] = datetime.now().strftime("%d-%m-%Y %I:%M %p")
                st.session_state.users_db[st.session_state.current_user]["result"] = result_text

            st.session_state.selected_menu = "Visualization"
            st.rerun()


# ==============================
# VISUALIZATION PAGE
# ==============================
elif st.session_state.logged_in and st.session_state.selected_menu == "Visualization":
    st.markdown("<div class='accent-line'></div>", unsafe_allow_html=True)
    st.title("Prediction Result and Visualization")

    if not st.session_state.prediction_done or st.session_state.latest_input is None:
        st.info("Please run prediction first.")
    else:
        data = st.session_state.latest_input
        result_text = st.session_state.latest_result
        probability = st.session_state.latest_probability

        glucose = data["Glucose"].values[0]
        bmi = data["BMI"].values[0]
        bp = data["BloodPressure"].values[0]
        insulin = data["Insulin"].values[0]
        age = data["Age"].values[0]
        dpf = data["DiabetesPedigreeFunction"].values[0]
        skin = data["SkinThickness"].values[0]

        if "High" in result_text:
            st.error(f"⚠️ {result_text}")
            st.markdown("<span class='risk-red'>HIGH RISK</span>", unsafe_allow_html=True)
            prediction_value = 1
        else:
            st.success(f"✅ {result_text}")
            st.markdown("<span class='risk-green'>LOW RISK</span>", unsafe_allow_html=True)
            prediction_value = 0

        if probability is not None:
            st.metric("Risk Probability", f"{probability * 100:.2f}%")

        st.info(get_risk_reason(glucose, bmi, insulin, bp, prediction_value))

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Glucose", "BMI", "Blood Pressure", "Insulin", "Overall"])

        with tab1:
            fig, ax = plt.subplots(figsize=(5.5, 3.2))
            ranges = ["Normal", "Prediabetic", "High", "Patient"]
            values = [99, 125, 160, glucose]
            ax.plot(ranges, values, marker="o", linewidth=2.5)
            ax.axhline(100, linestyle="--", alpha=0.5)
            ax.axhline(126, linestyle="--", alpha=0.5)
            ax.set_title("Glucose Risk Range")
            ax.set_ylabel("mg/dL")
            ax.grid(alpha=0.25)
            st.pyplot(fig, use_container_width=False)

        with tab2:
            fig, ax = plt.subplots(figsize=(5.5, 3.2))
            labels = ["Underweight", "Normal", "Overweight", "Obese", "Patient"]
            values = [18.5, 24.9, 29.9, 35, bmi]
            ax.plot(labels, values, marker="s", linewidth=2.2)
            ax.set_title("BMI Category Comparison")
            ax.set_ylabel("BMI")
            ax.grid(alpha=0.25)
            plt.xticks(rotation=20)
            st.pyplot(fig, use_container_width=False)

        with tab3:
            fig, ax = plt.subplots(figsize=(5.5, 3.2))
            ax.barh(["Patient BP"], [bp])
            ax.axvline(80, linestyle="--", label="Normal limit")
            ax.set_title("Blood Pressure Level")
            ax.set_xlabel("mm Hg")
            ax.legend()
            st.pyplot(fig, use_container_width=False)

        with tab4:
            fig, ax = plt.subplots(figsize=(5.5, 3.2))
            ax.scatter([glucose], [insulin], s=120)
            ax.set_title("Insulin vs Glucose")
            ax.set_xlabel("Glucose")
            ax.set_ylabel("Insulin")
            ax.grid(alpha=0.25)
            st.pyplot(fig, use_container_width=False)

        with tab5:
            fig, ax = plt.subplots(figsize=(5.8, 3.4))
            labels = ["Glucose", "BP", "Skin", "Insulin", "BMI", "DPF", "Age"]
            values = [glucose, bp, skin, insulin, bmi, dpf, age]
            ax.plot(labels, values, marker="o", linewidth=2.5)
            ax.fill_between(labels, values, alpha=0.15)
            ax.set_title("Overall Health Profile")
            ax.set_ylabel("Values")
            ax.grid(alpha=0.25)
            plt.xticks(rotation=20)
            st.pyplot(fig, use_container_width=False)

        st.markdown("### Medical Advice")
        advice = get_medical_advice(glucose, bmi, insulin, bp, prediction_value)
        for item in advice:
            st.write(f"• {item}")

        st.markdown("---")
        st.markdown("### Report Share")

        role = current_role()

        if role == "Doctor" and st.session_state.selected_patient_id:
            patients = get_doctor_patients()
            patient = patients.get(st.session_state.selected_patient_id, {})
            patient_name_report = patient.get("name", "Unknown Patient")
            patient_id_report = st.session_state.selected_patient_id
            patient_photo_report = patient.get("photo")
            doctor_report = True
        else:
            user = get_current_user()
            patient_name_report = user.get("full_name", "Patient")
            patient_id_report = "SELF"
            patient_photo_report = None
            doctor_report = False

        if st.button("Generate Report"):
            report_path = generate_patient_report(
                patient_name_report,
                patient_id_report,
                result_text,
                data,
                patient_photo_report,
                doctor_report
            )

            with open(report_path, "rb") as f:
                pdf_bytes = f.read()

            st.download_button(
                label="⬇️ Download Report",
                data=pdf_bytes,
                file_name=f"{patient_name_report}_diabetes_report.pdf",
                mime="application/pdf"
            )

            message = quote(
                f"GlucoTrack Diabetes Report\n"
                f"Patient: {patient_name_report}\n"
                f"Result: {result_text}\n"
                f"Risk Reason: {get_risk_reason(glucose, bmi, insulin, bp, prediction_value)}"
            )

            st.markdown(
                f"<a class='whatsapp-btn' href='https://wa.me/?text={message}' target='_blank'>📲 Share on WhatsApp</a>",
                unsafe_allow_html=True
            )


# ==============================
# ABOUT PAGE
# ==============================
elif st.session_state.logged_in and st.session_state.selected_menu == "About":
    st.markdown("<div class='accent-line'></div>", unsafe_allow_html=True)
    st.title("About GlucoTrack")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class='mini-card'>
            <h4>Fast Prediction</h4>
            <p>It gives risk output quickly after entering patient values.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='mini-card'>
            <h4>8 Main Inputs</h4>
            <p>Uses glucose, BMI, age, insulin, blood pressure and more.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class='mini-card'>
            <h4>Report Generate</h4>
            <p>Creates a clean PDF report for patient records.</p>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class='mini-card'>
            <h4>Doctor Dashboard</h4>
            <p>Doctors can enroll and review their own patients.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='main-card'>
        <p>
        GlucoTrack is a diabetes risk prediction dashboard made for project and demo purposes.
        It helps users understand possible risk from clinical input values.
        It is not a replacement for medical diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)
