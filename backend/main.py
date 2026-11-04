import os
import sqlite3
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
import tempfile

app = FastAPI(title="SKILLSYNC Backend", version="2.5")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SQLite Production Database
DB_FILE = os.path.join(tempfile.gettempdir(), "skillsync_production.db")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            college TEXT,
            year TEXT,
            department TEXT,
            match_score REAL,
            skill_gap REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Load Lightweight Sentence Transformer AI Model
print("Loading AI model for semantic analysis...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Industry Benchmarks & Skill Profiles
INDUSTRY_PROFILES = {
    "CSE / IT": {
        "benchmark": "Software engineer proficient in Python, SQL, data structures, algorithms, FastAPI backend development, Git version control, and machine learning model deployment.",
        "skills": ["Python", "SQL", "Data Structures", "FastAPI", "Git", "Machine Learning"]
    },
    "ECE": {
        "benchmark": "Embedded systems and electronics engineer skilled in C, C++, IoT architecture, circuit design, microcontrollers, Python automation, and signal processing.",
        "skills": ["C", "C++", "IoT", "Embedded Systems", "Python", "Circuit Design"]
    },
    "MECH": {
        "benchmark": "Mechanical design engineer experienced in CAD, SolidWorks, thermodynamics, manufacturing processes, automation, and Python data analysis.",
        "skills": ["CAD", "SolidWorks", "Thermodynamics", "Automation", "Python", "Manufacturing"]
    },
    "CIVIL": {
        "benchmark": "Civil structural engineer skilled in AutoCAD, STAAD.Pro, surveying, project management, construction materials, and structural analysis.",
        "skills": ["AutoCAD", "STAAD.Pro", "Surveying", "Project Management", "Construction", "Structural Analysis"]
    }
}

# Dynamic YouTube API Course Fetcher
def fetch_dynamic_youtube_course(skill_name):
    api_key = "my api key"  # Replace with your actual copied Google Cloud API key
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": f"{skill_name} full course tutorial",
        "type": "video",
        "maxResults": 1,
        "key": api_key
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        # Extract the top live video result from YouTube
        if "items" in data and len(data["items"]) > 0:
            item = data["items"][0]
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            return {
                "topic": title,
                "platform": "YouTube (Live API)",
                "link": f"https://www.youtube.com/watch?v={video_id}"
            }
    except Exception as e:
        print(f"Error fetching live YouTube video for {skill_name}: {e}")
    
    # Fallback search URL if API call fails or quota is exceeded
    return {
        "topic": f"Mastering {skill_name} Fundamentals",
        "platform": "YouTube",
        "link": f"https://www.youtube.com/results?search_query={skill_name}+full+course"
    }

@app.post("/api/analyze-resume")
async def analyze_resume(
    name: str = Form(...),
    college: str = Form(...),
    year: str = Form(...),
    department: str = Form(...),
    project_text: str = Form(default=""),
    file: UploadFile = File(default=None)
):
    extracted_text = project_text
    
    # Extract text from uploaded PDF resume if provided
    if file is not None:
        try:
            contents = await file.read()
            pdf_reader = PdfReader(io.BytesIO(contents))
            pdf_text = ""
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    pdf_text += text + "\n"
            if pdf_text.strip():
                extracted_text += "\n" + pdf_text
        except Exception as e:
            print(f"PDF extraction error: {e}")

    # Fetch corresponding industry benchmark profile
    profile = INDUSTRY_PROFILES.get(department, INDUSTRY_PROFILES["CSE / IT"])
    benchmark_text = profile["benchmark"]
    required_skills = profile["skills"]

    # Skill Gap & Matching Logic
    text_lower = extracted_text.lower()
    matched_skills = [skill for skill in required_skills if skill.lower() in text_lower]
    missing_skills = [skill for skill in required_skills if skill.lower() not in text_lower]

    skill_match_ratio = len(matched_skills) / len(required_skills) if required_skills else 0.0

    # AI Semantic Similarity Scoring
    try:
        emb1 = model.encode(extracted_text if extracted_text.strip() else benchmark_text, convert_to_tensor=True)
        emb2 = model.encode(benchmark_text, convert_to_tensor=True)
        semantic_score = float(util.cos_sim(emb1, emb2)[0][0])
    except Exception:
        semantic_score = 0.5

    # Hybrid Scoring Computation
    final_score = round(min(95.0, max(35.0, (skill_match_ratio * 70) + (max(0, semantic_score) * 20) + 10)), 1)
    skill_gap = round(max(5.0, 100.0 - final_score), 1)

    # Fetch Dynamic Recommendations for Missing Skills via YouTube API (Flattened)
    recommended_courses = []
    for skill in missing_skills:
        course_info = fetch_dynamic_youtube_course(skill)
        recommended_courses.append({
            "skill": skill,
            "topic": course_info["topic"],
            "platform": course_info["platform"],
            "link": course_info["link"]
        })

    # Persistently Log Results into SQLite Database
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO analytics (name, college, year, department, match_score, skill_gap)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, college, year, department, final_score, skill_gap))
        conn.commit()
        conn.close()
    except Exception as db_err:
        print(f"Database logging error: {db_err}")

    return {
        "status": "success",
        "name": name,
        "department": department,
        "match_score": final_score,
        "skill_gap": skill_gap,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "recommendations": recommended_courses
    }

@app.get("/")
def read_root():
    return {"message": "SKILLSYNC AI Backend is running successfully!"}