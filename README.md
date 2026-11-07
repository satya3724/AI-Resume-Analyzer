
# AI Resume Analyzer

A Flask web app that analyzes a resume (PDF) against a job description.

## Features
- Resume PDF upload and text extraction (pdfplumber)
- Skills detection from a skills database
- Resume score based on section completeness
- Job description matching: Skill Match % with matched and missing skills
- TF-IDF cosine similarity between resume and job description
- Improvement suggestions for missing skills

## Tech Stack
Python, Flask, pdfplumber, scikit-learn, HTML/CSS

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python app.py`
3. Open `http://127.0.0.1:5001` in your browser

## Screenshots
![Home](screenshots/home.png)
![Results 1](screenshots/results1.png)
![Results 2](screenshots/results2.png)

## Limitations and Future Work
- Skill detection is keyword-based (fixed skills list)
- Only PDF resumes supported (DOCX planned)
- Planned: LLM-based feedback, larger skills database
