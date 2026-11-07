import re
from flask import Flask, request, render_template_string
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

SKILLS_DB = [
    "python", "c", "embedded c", "c++", "java", "sql", "matlab",
    "arduino", "esp32", "raspberry pi", "sensor interfacing",
    "gps", "gsm", "iot", "pspice", "oscilloscope",
    "pcb", "circuit troubleshooting", "html", "css", "javascript",
    "flask", "machine learning", "data analysis", "git",
    "stm32", "rtos", "i2c", "spi", "uart", "mqtt", "vhdl", "verilog",
    "autocad", "labview", "simulink"
]

TIPS = {
    "stm32": "STM32 microcontrollers nerchukondi (STM32CubeIDE free).",
    "rtos": "FreeRTOS basics nerchukondi: tasks, queues, semaphores.",
    "i2c": "I2C protocol practice cheyandi (Arduino tho sensor interface cheyandi).",
    "spi": "SPI protocol practice cheyandi (SD card or display module tho).",
    "uart": "UART communication practice cheyandi (GPS/GSM modules tho already touch undi).",
    "mqtt": "MQTT nerchukondi, mee ESP32 project ki MQTT add cheyochu.",
    "python": "Python projects add cheyandi.",
    "sql": "SQL basics practice cheyandi.",
    "pcb": "KiCad tho simple PCB design cheyandi.",
    "vhdl": "VHDL basics nerchukondi.",
    "verilog": "Verilog basics nerchukondi.",
    "labview": "LabVIEW basics nerchukondi.",
    "simulink": "MATLAB Simulink tutorials chudandi.",
    "git": "Git and GitHub nerchukondi, projects upload cheyandi.",
}

SECTIONS = {
    "Summary": ["summary", "objective"],
    "Skills": ["skills"],
    "Education": ["education"],
    "Projects": ["projects"],
    "Experience / Internship": ["internship", "experience"],
    "Certifications": ["certification"],
}

PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Resume Analyzer</title>
<style>
 body{font-family:Segoe UI,Arial,sans-serif;background:#f0f4f8;margin:0;color:#222}
 .header{background:linear-gradient(135deg,#4f46e5,#7c3aed);color:white;padding:30px 20px;text-align:center}
 .header h1{margin:0;font-size:32px}
 .header p{margin:8px 0 0;opacity:.9}
 .container{max-width:850px;margin:25px auto;padding:0 15px}
 .card{background:white;border-radius:14px;padding:22px;margin-bottom:20px;box-shadow:0 2px 10px rgba(0,0,0,.08)}
 .card h3{margin-top:0}
 textarea{width:100%;box-sizing:border-box;padding:10px;border:1px solid #ccc;border-radius:8px;font-family:inherit}
 button{background:#4f46e5;color:white;border:none;padding:12px 28px;border-radius:8px;font-size:16px;cursor:pointer}
 button:hover{background:#4338ca}
 .tag{display:inline-block;padding:5px 12px;margin:3px;border-radius:20px;font-size:14px}
 .green{background:#d1fae5;color:#065f46}
 .red{background:#fee2e2;color:#991b1b}
 .bar{background:#e5e7eb;border-radius:20px;overflow:hidden}
 .fill{color:white;text-align:center;padding:6px 0;min-width:45px;border-radius:20px}
 .blue{background:#3b82f6}
 .ok{background:#10b981}
 .msg{color:#065f46;font-weight:bold}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:6px}
 pre{white-space:pre-wrap;background:#f4f4f4;padding:12px;border-radius:8px;font-size:13px}
</style>
</head>
<body>
  <div class="header">
    <h1>AI Resume Analyzer</h1>
    <p>Upload your resume, paste a job description, get instant feedback</p>
  </div>

  <div class="container">
    <div class="card">
      <form action="/" method="POST" enctype="multipart/form-data">
        <h3>1. Upload Resume (PDF)</h3>
        <input type="file" name="resume" accept=".pdf" required>
        <h3>2. Paste Job Description</h3>
        <textarea name="jd" rows="7" placeholder="Job description ikkada paste cheyandi...">{{ jd or '' }}</textarea>
        <br><br>
        <button type="submit">Analyze Resume</button>
      </form>
    </div>

    {% if message %}
      <div class="card"><span class="msg">{{ message }}</span></div>
    {% endif %}

    {% if score is not none %}
      <div class="card">
        <h3>Resume Score: {{ score }}/100</h3>
        <div class="bar"><div class="fill ok" style="width:{{ score }}%">{{ score }}%</div></div>
        <br>
        <div class="grid">
          {% for name, present in sections.items() %}
            <div>{{ '✅' if present else '❌' }} {{ name }}</div>
          {% endfor %}
        </div>
      </div>
    {% endif %}

    {% if skills is not none %}
      <div class="card">
        <h3>Skills in Resume ({{ skills|length }})</h3>
        {% for s in skills %}<span class="tag green">{{ s }}</span>{% endfor %}
      </div>
    {% endif %}

    {% if similarity is not none %}
      <div class="card">
        <h3>Text Similarity (TF-IDF): {{ similarity }}%</h3>
        <div class="bar"><div class="fill blue" style="width:{{ similarity }}%">{{ similarity }}%</div></div>
      </div>
    {% endif %}

    {% if match_percent is not none %}
      <div class="card">
        <h3>Skill Match: {{ match_percent }}%</h3>
        <div class="bar"><div class="fill ok" style="width:{{ match_percent }}%">{{ match_percent }}%</div></div>
        <h4>Matched Skills ({{ matched|length }})</h4>
        {% for s in matched %}<span class="tag green">{{ s }}</span>{% else %}<p>Emi match avvaledu.</p>{% endfor %}
        <h4>Missing Skills ({{ missing|length }})</h4>
        {% for s in missing %}<span class="tag red">{{ s }}</span>{% else %}<p>Missing skills levu. 🎉</p>{% endfor %}
      </div>

      {% if suggestions %}
      <div class="card">
        <h3>Suggestions</h3>
        <ul>{% for tip in suggestions %}<li>{{ tip }}</li>{% endfor %}</ul>
      </div>
      {% endif %}
    {% elif jd_given %}
      <div class="card" style="color:#b45309;">JD lo mee skills list lo unna skills emi dorakaledu.</div>
    {% endif %}

    {% if text %}
      <div class="card">
        <details>
          <summary><b>Extracted Text (click to view)</b></summary>
          <pre>{{ text }}</pre>
        </details>
      </div>
    {% endif %}
  </div>
</body>
</html>
"""

def extract_skills(text):
    text_lower = text.lower()
    found = []
    for skill in SKILLS_DB:
        if len(skill) <= 3:
            if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
                found.append(skill)
        elif skill in text_lower:
            found.append(skill)
    return found

def check_sections(text):
    text_lower = text.lower()
    return {name: any(k in text_lower for k in kws) for name, kws in SECTIONS.items()}

def tfidf_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(float(score) * 100)

@app.route("/", methods=["GET", "POST"])
def home():
    message = text = skills = score = similarity = None
    sections = {}
    jd = ""
    jd_given = False
    matched, missing, suggestions = [], [], []
    match_percent = None

    if request.method == "POST":
        jd = request.form.get("jd", "").strip()
        file = request.files.get("resume")
        if file and file.filename != "":
            try:
                with pdfplumber.open(file) as pdf:
                    pages = [page.extract_text() or "" for page in pdf.pages]
                    page_count = len(pdf.pages)
                text = "\n".join(pages).strip()
                message = f"Resume received: {file.filename} ({page_count} pages)"
                if text:
                    skills = extract_skills(text)
                    sections = check_sections(text)
                    score = round(sum(sections.values()) / len(sections) * 100)
                    if jd:
                        jd_given = True
                        similarity = tfidf_similarity(text, jd)
                        jd_skills = extract_skills(jd)
                        matched = [s for s in jd_skills if s in skills]
                        missing = [s for s in jd_skills if s not in skills]
                        suggestions = [TIPS[s] for s in missing if s in TIPS]
                        if jd_skills:
                            match_percent = round(len(matched) / len(jd_skills) * 100)
                else:
                    message += " - text extract avvaledu (scanned PDF avvachu)"
            except Exception as e:
                message = f"PDF read cheyadam lo error: {e}"
        else:
            message = "No file selected."

    return render_template_string(
        PAGE, message=message, text=text, skills=skills, jd=jd,
        jd_given=jd_given, matched=matched, missing=missing,
        match_percent=match_percent, score=score, sections=sections,
        suggestions=suggestions, similarity=similarity
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)