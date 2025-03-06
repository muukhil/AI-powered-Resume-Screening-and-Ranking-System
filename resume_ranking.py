import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Professionally written job descriptions
job_descriptions = {
    "Full Stack Developer": """We are seeking a highly skilled Full Stack Developer responsible for 
    designing, developing, and maintaining both front-end and back-end components of web applications. 
    The ideal candidate should have expertise in modern frameworks, databases, and cloud technologies, 
    with a strong understanding of RESTful APIs, microservices architecture, and DevOps practices.""",

    "Software Testing Engineer": """We are hiring a Software Testing Engineer to ensure the delivery of 
    high-quality software products. The candidate will be responsible for designing test plans, executing 
    automated and manual test cases, identifying defects, and collaborating with developers to resolve issues. 
    Knowledge of testing frameworks, debugging tools, and CI/CD pipelines is required.""",

    "Data Scientist": """We are looking for a Data Scientist to analyze large and complex datasets, extract 
    meaningful insights, and drive data-driven decision-making. The candidate should be proficient in machine 
    learning, deep learning, statistical modeling, and big data technologies. Expertise in Python, R, SQL, 
    and cloud-based AI services is essential.""",

    "Data Analyst": """We are seeking a Data Analyst to collect, process, and interpret data to generate actionable 
    business insights. The candidate should have expertise in data visualization, statistical analysis, 
    and predictive modeling using tools like Python, SQL, Power BI, or Tableau. Strong analytical skills 
    and the ability to communicate findings effectively are essential.""",

    "Mobile App Developer": """We are looking for a Mobile App Developer to design, develop, and optimize 
    high-performance applications for Android and iOS platforms. The ideal candidate should be experienced 
    in mobile UI/UX, cross-platform development (React Native, Flutter), API integration, and performance 
    optimization techniques.""",
}

# Function to extract text from PDF resumes
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""  # Handle cases where text extraction fails
    return text.strip()

# Function to rank resumes against job descriptions
def rank_resumes(resumes, job_description):
    results = []
    for resume_name, resume_text in resumes.items():
        documents = [job_description, resume_text]  # Compare resume directly with selected job description
        vectorizer = TfidfVectorizer().fit_transform(documents)
        vectors = vectorizer.toarray()

        # Compute cosine similarity
        score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]  # Similarity between job and resume
        results.append((resume_name, score))

    return results

# Streamlit UI
st.title("AI Resume Screening & Candidate Ranking System")

# Job description selection/input
st.header("Job Description")
job_options = ["Select a predefined job role"] + list(job_descriptions.keys()) + ["Enter a custom job description"]
selected_job = st.selectbox("Choose a job role or enter a custom job description:", job_options)

custom_job_description = ""
if selected_job == "Enter a custom job description":
    custom_job_description = st.text_area("Enter the custom job description here:")

# File uploader for resumes
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files and (selected_job != "Select a predefined job role" or custom_job_description):
    st.header("Ranking Resumes")

    # Extract text from resumes
    resumes = {file.name: extract_text_from_pdf(file) for file in uploaded_files}

    # Get the job description (predefined or custom)
    job_description = custom_job_description if selected_job == "Enter a custom job description" else job_descriptions[selected_job]

    # Rank resumes
    ranked_results = rank_resumes(resumes, job_description)

    # Convert to DataFrame and display results
    results_df = pd.DataFrame(ranked_results, columns=["Resume", "Score"])
    results_df = results_df.sort_values(by="Score", ascending=False)

    st.write(results_df)
