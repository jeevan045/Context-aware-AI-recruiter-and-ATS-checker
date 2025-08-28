Context-Aware AI Recruiter

This project is a smart recruitment tool that moves beyond traditional keyword matching. It leverages Natural Language Processing (NLP) to perform a deep, contextual analysis of a candidate's resume against a job description, providing a more accurate and insightful measure of their suitability for the role.

The application features a clean, modern web interface with drag-and-drop file uploads, and is powered by a Python backend that handles all the complex analysis.

✨ Features
Semantic Matching: Instead of just checking for keywords, the AI understands the meaning and context behind the text in both the resume and the job description using sentence-transformer models.

Experience Extraction: Intelligently parses the resume text to identify and extract the candidate's years of professional experience.

Skill Gap Analysis: Automatically compares the skills required in the job description against those present in the resume, generating lists of both Matched Skills and Missing Skills.

Holistic Scoring: Calculates a comprehensive Overall Match Score based on a weighted combination of three key factors, which are clearly displayed in the results:

Semantic Similarity: How well the context of the resume matches the job description.

Skills Coverage: The percentage of required skills found in the resume.

Years of Experience: The candidate's level of experience.

Interactive Web UI: A user-friendly interface for pasting job descriptions, uploading resumes (including drag-and-drop), and viewing the detailed analysis.

🛠️ Tech Stack
Backend: Python, Flask

Frontend: HTML, Tailwind CSS, JavaScript

Core AI/NLP: sentence-transformers for semantic embeddings, spacy for NLP tasks, and PyPDF2/pdfplumber for PDF parsing.
