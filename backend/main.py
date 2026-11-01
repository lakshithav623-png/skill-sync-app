from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sentence_transformers import SentenceTransformer, util
from pypdf import PdfReader
import io
import os
import sqlite3

app = FastAPI(title="SKILLSYNC Final Production Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading AI Embedding Model (all-MiniLM-L6-v2)...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

INDUSTRY_PROFILES = {
    "CSE / IT": {
        "benchmark": "Expert in Python programming, data structures, algorithms, FastAPI backend development, database management with SQL, and React frontend engineering.",
        "skills": ["Python", "Data Structures", "FastAPI", "SQL", "React", "Git"],
        "company": "Zoho / TCS / Infosys / Google",
        "role": "Full Stack / AI Software Engineer"
    },
    "ECE": {
        "benchmark": "Proficient in Embedded C, Arduino microcontrollers, VLSI design, PCB circuit schematics, and IoT sensor integration.",
        "skills": ["Embedded C", "Arduino", "VLSI", "PCB Design", "IoT", "Microcontrollers"],
        "company": "Qualcomm / Bosch / ISRO",
        "role": "Embedded Systems Engineer"
    },
    "MECH": {
        "benchmark": "Skilled in AutoCAD mechanical drafting, SolidWorks 3D modeling, ANSYS finite element thermal analysis, and GD&T precision manufacturing.",
        "skills": ["AutoCAD", "SolidWorks", "ANSYS", "GD&T", "Thermal Analysis", "Machining"],
        "company": "Hyundai / Ashok Leyland / L&T",
        "role": "CAD & Design Engineer"
    },
    "CIVIL": {
        "benchmark": "Expert in AutoCAD structural blueprints, STAAD.Pro analysis, Revit 3D building information modeling, and advanced concrete technology.",
        "skills": ["AutoCAD", "STAAD.Pro", "Revit", "Structural Analysis", "Surveying", "Concrete Technology"],
        "company": "L&T Construction / Shapoorji Pallonji",
        "role": "Structural Site Engineer"
    }
}

COURSE_MAPPING = {
    "python": {"topic": "Mastering Python Programming & Automation", "platform": "YouTube / NPTEL", "link": "https://www.youtube.com/results?search_query=python+full+course+for+beginners"},
    "data structures": {"topic": "Data Structures & Algorithms Masterclass", "platform": "Coursera / YouTube", "link": "https://www.youtube.com/results?search_query=data+structures+algorithms+full+course"},
    "fastapi": {"topic": "FastAPI Backend Web Development", "platform": "YouTube", "link": "https://www.youtube.com/results?search_query=fastapi+tutorial+python"},
    "sql": {"topic": "SQL Databases & Query Optimization", "platform": "Coursera", "link": "https://www.coursera.org/courses?query=sql"},
    "react": {"topic": "React.js Frontend Web Development", "platform": "YouTube", "link": "https://www.youtube.com/results?search_query=react.js+full+course"},
    "git": {"topic": "Git & GitHub Version Control Workflow", "platform": "YouTube", "link": "https://www.youtube.com/results?search_query=git+github+tutorial"},
    "embedded c": {"topic": "Embedded C Programming for Microcontrollers", "platform": "YouTube", "link": "https://www.youtube.com/results?search_query=embedded+c+programming+tutorial"},
    "arduino": {"topic": "Arduino & IoT Prototyping", "platform": "Coursera / YouTube", "link": "https://www.youtube.com/results?search_query=arduino+iot+projects"},
    "autocad": {"topic": "AutoCAD Professional Drafting & Blueprint Design", "platform": "Coursera", "link": "https://www.coursera.org/courses?query=autocad"},
    "solidworks": {"topic": "SolidWorks 3D Modeling & Mechanical Design", "platform": "YouTube", "link": "https://www.youtube.com/results?search_query=solidworks+tutorial+for+beginners"},
    "staad.pro": {"topic": "STAAD.Pro Structural Analysis Masterclass", "platform": "YouTube", "link": "https://www.youtube.com/results?search_query=staad.pro+tutorial"}
}

def init_db():
    conn = sqlite3.connect('skillsync_production.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS production_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            department TEXT,
            score REAL,
            gap REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.get("/")
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "../frontend/index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"error": "frontend/index.html not found"}

@app.post("/api/analyze-resume")
async def analyze_resume(
    name: str = Form(...),
    college: str = Form(...),
    year: str = Form(...),
    department: str = Form(...),
    project_text: str = Form(""),
    resume: UploadFile = File(None)
):
    resume_text = project_text
    if resume:
        contents = await resume.read()
        try:
            reader = PdfReader(io.BytesIO(contents))
            for page in reader.pages:
                resume_text += " " + (page.extract_text() or "")
        except Exception:
            pass

    if not resume_text.strip():
        resume_text = f"Student in {department} at {college}."

    profile_data = INDUSTRY_PROFILES.get(department, INDUSTRY_PROFILES["CSE / IT"])
    benchmark_text = profile_data["benchmark"]

    student_vector = embed_model.encode(resume_text, convert_to_tensor=True)
    benchmark_vector = embed_model.encode(benchmark_text, convert_to_tensor=True)

    text_lower = resume_text.lower()
    matched_skills = [s for s in profile_data["skills"] if s.lower() in text_lower]
    missing_skills = [s for s in profile_data["skills"] if s.lower() not in text_lower]

    if not matched_skills and not missing_skills:
        matched_skills = profile_data["skills"][:2]
        missing_skills = profile_data["skills"][2:]

    total_skills = len(profile_data["skills"])
    matched_count = len(matched_skills)
    skill_match_ratio = matched_count / total_skills if total_skills > 0 else 1.0

    skill_score_component = skill_match_ratio * 70.0
    semantic_similarity = util.cos_sim(student_vector, benchmark_vector).item()
    semantic_bonus = max(0.0, min(semantic_similarity * 20.0, 20.0))
    base_score = 10.0

    resume_score = round(min(skill_score_component + semantic_bonus + base_score, 95.0), 1)
    required_benchmark = 85.0 if department in ["CSE / IT", "ECE"] else 80.0
    resume_gap = round(max(0.0, required_benchmark - resume_score), 1)

    recommendations = []
    for skill in missing_skills:
        if skill.lower() in COURSE_MAPPING:
            recommendations.append(COURSE_MAPPING[skill.lower()])

    if not recommendations:
        recommendations.append({
            "topic": f"Advanced Professional Training & Capstone in {department}",
            "platform": "NPTEL / Coursera",
            "link": "https://nptel.ac.in"
        })

    conn = sqlite3.connect('skillsync_production.db')
    c = conn.cursor()
    c.execute("INSERT INTO production_records (name, department, score, gap) VALUES (?, ?, ?, ?)",
              (name, department, resume_score, resume_gap))
    conn.commit()
    conn.close()

    return {
        "status": "success",
        "name": name,
        "department": department,
        "resume_score": resume_score,
        "resume_gap": resume_gap,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "target_company": profile_data["company"],
        "recommended_role": profile_data["role"],
        "recommendations": recommendations
    }