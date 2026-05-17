import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

MODEL_PATH = "career_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
FEEDBACK_PATH = "feedback.csv"
FEATURES = [
    "Coding_and_Algorithms",
    "UI_and_Visual_Design",
    "Data_and_Analytics",
    "Math_and_Predictive_Modeling",
    "Infrastructure_and_Automation",
    "Security_and_Networking",
    "Business_and_Product_Strategy",
    "System_Architecture_and_APIs",
]

EMOJI_FEEDBACK_OPTIONS = [
    {"emoji": "😡", "label": "Very Uninterested", "value": 1},
    {"emoji": "😕", "label": "Low Interest", "value": 2},
    {"emoji": "😐", "label": "Neutral", "value": 3},
    {"emoji": "🙂", "label": "Interested", "value": 4},
    {"emoji": "😍", "label": "Very Excited", "value": 5},
]

FEEDBACK_COLUMNS = [
    "timestamp",
    "predicted_role",
    "interest_rating_numeric",
    "interest_rating_emoji",
]


def load_model(model_path: str, encoder_path: str):
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    if not os.path.exists(encoder_path):
        st.error(f"Label encoder file not found: {encoder_path}")
        st.stop()

    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    with open(encoder_path, "rb") as encoder_file:
        label_encoder = pickle.load(encoder_file)

    return model, label_encoder


def format_strengths(inputs: dict, top_n: int = 3):
    sorted_strengths = sorted(inputs.items(), key=lambda item: item[1], reverse=True)
    return sorted_strengths[:top_n]


def build_roadmap(role: str) -> dict:
    roadmap = {
        "AI Engineer": {
            "Beginner": [
                "Build Python fundamentals and data pipelines.",
                "Complete introductory ML courses on Coursera or Udacity.",
            ],
            "Intermediate": [
                "Practice model training and experiment with scikit-learn.",
                "Build end-to-end AI projects using TensorFlow or PyTorch.",
            ],
            "Advanced": [
                "Deploy AI models to production with Kubernetes or cloud services.",
                "Study MLOps, model monitoring, and interpretability.",
            ],
            "Certifications": ["Google Cloud Professional ML Engineer", "AWS Certified Machine Learning - Specialty"],
            "Projects": [
                "Real-time image classification app.",
                "Predictive analytics dashboard for customer churn.",
            ],
            "Interview": [
                "Review ML algorithms, bias-variance, and feature engineering.",
                "Practice system design for scalable AI solutions.",
            ],
        },
        "Cloud Engineer": {
            "Beginner": [
                "Learn cloud basics and infrastructure concepts.",
                "Practice with AWS, Azure, or GCP free tiers.",
            ],
            "Intermediate": [
                "Build CI/CD pipelines and infrastructure-as-code templates.",
                "Deploy containerized apps with Docker and Kubernetes.",
            ],
            "Advanced": [
                "Design resilient multi-cloud architectures.",
                "Optimize cost and security for cloud workloads.",
            ],
            "Certifications": ["AWS Certified Solutions Architect", "Azure Administrator Associate"],
            "Projects": [
                "Infrastructure automation with Terraform.",
                "High-availability cloud service deployment.",
            ],
            "Interview": [
                "Prepare cloud architecture scenarios and networking questions.",
                "Discuss disaster recovery and performance scaling.",
            ],
        },
        "Cybersecurity Analyst": {
            "Beginner": [
                "Understand network basics, firewalls, and encryption.",
                "Complete cybersecurity fundamentals and ethical hacking labs.",
            ],
            "Intermediate": [
                "Practice incident response and threat hunting exercises.",
                "Study penetration testing and security auditing tools.",
            ],
            "Advanced": [
                "Lead security architecture reviews and defense strategies.",
                "Build SOC workflows and vulnerability management programs.",
            ],
            "Certifications": ["CompTIA Security+", "Certified Ethical Hacker (CEH)"],
            "Projects": [
                "Simulate penetration testing on a lab network.",
                "Create a security incident response playbook.",
            ],
            "Interview": [
                "Practice threat modeling and malware analysis case studies.",
                "Explain risk assessment and mitigation processes.",
            ],
        },
        "Data Analyst": {
            "Beginner": [
                "Master Excel, SQL, and data visualization basics.",
                "Build reports and dashboards with Power BI or Tableau.",
            ],
            "Intermediate": [
                "Analyze business problems and deliver actionable insights.",
                "Practice analytics with real datasets and storytelling.",
            ],
            "Advanced": [
                "Lead analytics projects and define KPIs for stakeholders.",
                "Design automated reporting and dashboard pipelines.",
            ],
            "Certifications": ["Microsoft Certified: Power BI Data Analyst", "Google Data Analytics Professional"],
            "Projects": [
                "Sales performance dashboard for business leaders.",
                "Customer segmentation and insight report.",
            ],
            "Interview": [
                "Prepare case studies that show clear business impact.",
                "Practice SQL and dashboard explanation questions.",
            ],
        },
        "Data Scientist": {
            "Beginner": [
                "Learn Python for data science and statistical fundamentals.",
                "Complete introductory machine learning projects.",
            ],
            "Intermediate": [
                "Build end-to-end modeling pipelines and feature engineering.",
                "Apply more advanced algorithms like ensemble methods.",
            ],
            "Advanced": [
                "Deploy predictive systems and optimize model performance.",
                "Lead data science projects with measurable business impact.",
            ],
            "Certifications": ["IBM Data Science Professional Certificate", "Azure Data Scientist Associate"],
            "Projects": [
                "Churn prediction model for customer retention.",
                "Recommendation system for personalized experiences.",
            ],
            "Interview": [
                "Practice statistics questions and model evaluation techniques.",
                "Explain ML workflows and business use cases clearly.",
            ],
        },
        "DevOps Engineer": {
            "Beginner": [
                "Study Linux, shell scripting, and system administration.",
                "Learn source control and basic CI/CD tooling.",
            ],
            "Intermediate": [
                "Build deployment pipelines and container orchestration.",
                "Implement monitoring and logging for applications.",
            ],
            "Advanced": [
                "Design scalable infrastructure and automation at enterprise scale.",
                "Lead DevOps transformation and platform engineering adoption.",
            ],
            "Certifications": ["AWS Certified DevOps Engineer", "Docker Certified Associate"],
            "Projects": [
                "Automated deployment pipeline with Jenkins or GitHub Actions.",
                "Containerized microservices platform with Kubernetes.",
            ],
            "Interview": [
                "Practice deployment, rollback, and incident response scenarios.",
                "Discuss infrastructure-as-code and automation tradeoffs.",
            ],
        },
        "Mobile App Developer": {
            "Beginner": [
                "Learn mobile UI frameworks and app lifecycle basics.",
                "Build simple apps using Flutter, React Native, or native SDKs.",
            ],
            "Intermediate": [
                "Implement async data flows and responsive layouts.",
                "Add integrations with APIs and backend services.",
            ],
            "Advanced": [
                "Deliver polished apps with performance optimization.",
                "Lead mobile architecture and app release strategies.",
            ],
            "Certifications": ["Google Associate Android Developer", "Microsoft Certified: Power Platform App Maker"],
            "Projects": [
                "Task tracking or social networking mobile app.",
                "Mobile e-commerce or fitness tracker app.",
            ],
            "Interview": [
                "Practice UI/UX tradeoffs and app architecture choices.",
                "Explain mobile performance and platform compatibility.",
            ],
        },
        "Non-Tech": {
            "Beginner": [
                "Explore strengths in communication and product strategy.",
                "Research business roles like product operations or client success.",
            ],
            "Intermediate": [
                "Build skills in stakeholder collaboration and project coordination.",
                "Work on business analysis and service design projects.",
            ],
            "Advanced": [
                "Lead cross-functional teams and shape customer-driven strategy.",
                "Position yourself as a trusted advisor in the organization.",
            ],
            "Certifications": ["Certified Scrum Master", "Professional in Business Analysis (PMI-PBA)"],
            "Projects": [
                "Customer journey mapping and process improvement case study.",
                "Business operations improvement proposal.",
            ],
            "Interview": [
                "Practice communication, stakeholder handling, and problem solving.",
                "Explain how you align business outcomes with team goals.",
            ],
        },
        "Software Engineer": {
            "Beginner": [
                "Learn clean coding practices and software fundamentals.",
                "Solve programming challenges in Python or Java.",
            ],
            "Intermediate": [
                "Build full-stack applications and collaborate on code reviews.",
                "Practice design patterns and software architecture principles.",
            ],
            "Advanced": [
                "Lead large-scale systems design and engineering best practices.",
                "Mentor teams and shape product engineering strategy.",
            ],
            "Certifications": ["AWS Certified Developer", "Oracle Java Certification"],
            "Projects": [
                "Full-stack web application with API backend.",
                "Scalable service with database integration and tests.",
            ],
            "Interview": [
                "Practice coding challenges and system design conversations.",
                "Explain architecture decisions clearly and concisely.",
            ],
        },
        "UI/UX Designer": {
            "Beginner": [
                "Study user-centered design and visual communication.",
                "Practice wireframing and prototyping in Figma.",
            ],
            "Intermediate": [
                "Build high-fidelity user interfaces and usability tests.",
                "Develop design systems and interaction patterns.",
            ],
            "Advanced": [
                "Lead product design strategy and experience research.",
                "Create polished end-to-end user journeys.",
            ],
            "Certifications": ["Google UX Design Professional Certificate", "NN/g UX Certification"],
            "Projects": [
                "Redesign an app experience with a clickable prototype.",
                "Run a user research study and present insight-driven designs.",
            ],
            "Interview": [
                "Prepare case studies showing process and design thinking.",
                "Discuss how your work improved usability or conversion.",
            ],
        },
        "Web Developer": {
            "Beginner": [
                "Learn HTML, CSS, and JavaScript fundamentals.",
                "Build small responsive websites and landing pages.",
            ],
            "Intermediate": [
                "Create interactive web apps with modern frameworks.",
                "Practice frontend-backend integration and API usage.",
            ],
            "Advanced": [
                "Optimize performance, accessibility, and responsive design.",
                "Build production-ready web applications and developer workflows.",
            ],
            "Certifications": ["FreeCodeCamp Responsive Web Design", "Meta Front-End Developer Professional Certificate"],
            "Projects": [
                "Portfolio website with interactive features.",
                "Dynamic website with API-powered content.",
            ],
            "Interview": [
                "Prepare questions on responsive design and DOM manipulation.",
                "Explain how you build modern and accessible front-ends.",
            ],
        },
    }

    return roadmap.get(role, roadmap["Non-Tech"])


def build_dashboard_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #0d1117 0%, #111827 45%, #0f172a 100%);
            color: #f8fafc;
        }

        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 24px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(20px);
            padding: 2rem;
            margin-bottom: 1.5rem;
        }

        .glow-box {
            background: rgba(15, 23, 42, 0.95);
            border-radius: 24px;
            padding: 1.5rem;
            border: 1px solid rgba(59, 130, 246, 0.35);
            box-shadow: 0 0 30px rgba(59, 130, 246, 0.35);
        }

        .metric-card {
            border-radius: 20px;
            padding: 1.2rem;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(255,255,255,0.08);
        }

        .hover-card:hover {
            transform: translateY(-4px);
            transition: transform 0.2s ease-in-out;
        }

        .stButton>button {
            background: linear-gradient(135deg, #22d3ee 0%, #2563eb 100%);
            color: white;
            border: none;
            border-radius: 18px;
            min-height: 4rem;
            min-width: 4rem;
            padding: 0.8rem 1rem;
            font-size: 2.4rem;
            box-shadow: 0 16px 32px rgba(15, 23, 42, 0.35);
            transition: transform 0.18s ease, box-shadow 0.2s ease, background 0.2s ease;
        }

        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 18px 36px rgba(15, 23, 42, 0.45);
        }

        .feedback-selected-pill {
            display: inline-flex;
            justify-content: center;
            align-items: center;
            gap: 0.35rem;
            border: 1px solid rgba(56, 189, 248, 0.55);
            background: rgba(56, 189, 248, 0.12);
            color: #e0f2fe;
            border-radius: 999px;
            padding: 0.35rem 0.9rem;
            font-size: 0.95rem;
            margin-bottom: 0.6rem;
        }

        .selected-banner {
            margin-top: 1.2rem;
            background: rgba(15, 23, 42, 0.96);
            border: 1px solid rgba(56, 189, 248, 0.18);
            border-radius: 20px;
            padding: 1rem 1.2rem;
        }

        .selected-banner strong {
            color: #a5f3fc;
        }

        .emoji-summary {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 1.1rem;
            color: #f8fafc;
        }

        .emoji-summary .emoji-icon {
            font-size: 2.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def create_strength_charts(user_values: dict, confidence: float):
    strengths_df = pd.DataFrame(
        {
            "Skill": list(user_values.keys()),
            "Value": list(user_values.values()),
        }
    )

    radar = go.Figure(
        go.Scatterpolar(
            r=strengths_df["Value"],
            theta=strengths_df["Skill"],
            fill="toself",
            marker_color="#38bdf8",
            fillcolor="rgba(56, 189, 248, 0.3)",
        )
    )
    radar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        polar_bgcolor="#0f172a",
        font_color="#f8fafc",
        polar=dict(bgcolor="#111827", radialaxis=dict(range=[0, 10], visible=True)),
        margin=dict(l=20, r=20, t=30, b=20),
    )

    bar = px.bar(
        strengths_df,
        x="Skill",
        y="Value",
        color="Value",
        color_continuous_scale="Blues",
        range_y=[0, 10],
    )
    bar.update_layout(
        plot_bgcolor="#0d1117",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f8fafc",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    bar.update_xaxes(tickangle= -45)

    confidence_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#f8fafc"},
                "bar": {"color": "#38bdf8"},
                "bgcolor": "#111827",
                "borderwidth": 1,
                "bordercolor": "#252f3f",
            },
            number={"suffix": "%", "font": {"color": "#f8fafc"}},
            title={"text": "Prediction Confidence", "font": {"color": "#f8fafc"}},
        )
    )
    confidence_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    return radar, bar, confidence_gauge


def save_feedback(predicted_role: str, interest_rating_numeric: int, interest_rating_emoji: str):
    timestamp = datetime.now().isoformat()
    feedback = {
        "timestamp": timestamp,
        "predicted_role": predicted_role,
        "interest_rating_numeric": interest_rating_numeric,
        "interest_rating_emoji": interest_rating_emoji,
    }
    df = pd.DataFrame([feedback])
    try:
        if os.path.exists(FEEDBACK_PATH):
            df.to_csv(FEEDBACK_PATH, mode="a", index=False, header=False)
        else:
            df.to_csv(FEEDBACK_PATH, index=False)
    except OSError as error:
        st.error(f"Unable to save feedback: {error}")


def init_session_state():
    if "feedback_selected_value" not in st.session_state:
        st.session_state.feedback_selected_value = None
        st.session_state.feedback_selected_emoji = ""
        st.session_state.feedback_selected_label = ""
        st.session_state.feedback_last_saved = {}
        st.session_state.feedback_save_message = ""
    if "prediction" not in st.session_state:
        st.session_state.prediction = {}


def save_feedback_if_new(predicted_role: str, option: dict):
    last_saved = st.session_state.feedback_last_saved
    current_save = {
        "predicted_role": predicted_role,
        "value": option["value"],
        "emoji": option["emoji"],
    }

    if last_saved == current_save:
        st.session_state.feedback_save_message = "This reaction has already been recorded."
        return False

    save_feedback(predicted_role, option["value"], option["emoji"])
    st.session_state.feedback_last_saved = current_save
    st.session_state.feedback_save_message = (
        f"Saved your reaction: {option['emoji']} ({option['label']})."
    )
    return True


def render_feedback_section(predicted_role: str):
    st.markdown("## How do you feel about this career?")
    cols = st.columns(len(EMOJI_FEEDBACK_OPTIONS))

    for option, col in zip(EMOJI_FEEDBACK_OPTIONS, cols):
        with col:
            if st.session_state.feedback_selected_value == option["value"]:
                st.markdown(
                    "<div class='feedback-selected-pill'>Selected</div>",
                    unsafe_allow_html=True,
                )
            if st.button(option["emoji"], key=f"feedback_{option['value']}"):
                st.session_state.feedback_selected_value = option["value"]
                st.session_state.feedback_selected_emoji = option["emoji"]
                st.session_state.feedback_selected_label = option["label"]
                save_feedback_if_new(predicted_role, option)

    if st.session_state.feedback_selected_value:
        st.markdown(
            "<div class='selected-banner'>"
            f"<div class='emoji-summary'><span class='emoji-icon'>{st.session_state.feedback_selected_emoji}</span>"
            f"<strong>You selected</strong> {st.session_state.feedback_selected_label}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    if st.session_state.feedback_save_message:
        st.success(st.session_state.feedback_save_message)


def build_summary(role: str, confidence_score: float, strengths: list[str]):
    summary = f"### Career Recommendation\n"
    summary += f"**Predicted role:** {role}  \n"
    summary += f"**Confidence score:** {confidence_score:.0%}  \n"
    summary += "**Top strengths:** " + ", ".join([name for name, _ in strengths]) + "  \n"

    if confidence_score < 0.65:
        summary += "\nThis model is not highly confident in the recommendation. "
        summary += "You may be well-suited for a flexible career path like Non-Tech, or focus on improving the lower-rated strengths."
    else:
        summary += "\nThis result suggests a strong alignment with your selected technical strengths."

    return summary


def main():
    st.set_page_config(
        page_title="AI Career Path Finder",
        page_icon="🤖",
        layout="wide",
    )

    build_dashboard_style()

    with st.sidebar:
        st.markdown("# Career Path Finder")
        st.markdown("### Discover your best tech role")
        st.markdown(
            "Use the skills slider panel to input your strength and get a tailored role recommendation."
        )
        st.markdown("---")
        st.markdown("### Navigation")
        st.write("- Dashboard")
        st.write("- Prediction")
        st.write("- Roadmap")
        st.write("- Feedback")
        st.markdown("---")
        st.caption("Built for AI career guidance with modern UX.")

    st.markdown("# AI Career Path Finder")
    st.markdown(
        "#### Predict your next tech career based on your strengths in technical domains."
    )

    init_session_state()
    model, label_encoder = load_model(MODEL_PATH, LABEL_ENCODER_PATH)

    with st.container():
        left_col, right_col = st.columns([2, 1])

        with left_col:
            st.markdown("## Skill Input Panel")
            skill_values = {}
            for feature in FEATURES:
                skill_values[feature] = st.slider(
                    feature.replace("_", " "),
                    min_value=0,
                    max_value=10,
                    value=5,
                    step=1,
                )

            if st.button("Predict Career", key="predict_career"):
                input_vector = np.array([list(skill_values.values())])
                predicted_idx = model.predict(input_vector)[0]
                predicted_role = label_encoder.inverse_transform([predicted_idx])[0]
                confidence_score = 0.0
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_vector)[0]
                    confidence_score = float(np.max(proba))

                top_strengths = format_strengths(skill_values)
                st.session_state.prediction = {
                    "predicted_role": predicted_role,
                    "confidence_score": confidence_score,
                    "top_strengths": top_strengths,
                    "roadmap": build_roadmap(predicted_role),
                }
                st.session_state.feedback_selected_value = None
                st.session_state.feedback_selected_emoji = ""
                st.session_state.feedback_selected_label = ""
                st.session_state.feedback_last_saved = {}
                st.session_state.feedback_save_message = ""

        if st.session_state.prediction:
            prediction = st.session_state.prediction
            predicted_role = prediction["predicted_role"]
            confidence_score = prediction["confidence_score"]
            top_strengths = prediction["top_strengths"]
            roadmap = prediction["roadmap"]

            with right_col:
                st.markdown("## Prediction Result")
                st.markdown("<div class='glow-box'>", unsafe_allow_html=True)
                st.markdown(f"### {predicted_role}")
                st.markdown(f"**Confidence:** {confidence_score:.0%}")
                st.markdown("**Top strengths:**")
                for name, value in top_strengths:
                    st.markdown(f"- {name}: {value}")
                st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Prediction Summary"):
                st.markdown(build_summary(predicted_role, confidence_score, top_strengths))

            radar, bar, confidence_gauge = create_strength_charts(
                skill_values, confidence_score
            )
            st.plotly_chart(radar, use_container_width=True)
            st.plotly_chart(bar, use_container_width=True)
            st.plotly_chart(confidence_gauge, use_container_width=True)

            st.markdown("## Personalized Roadmap")
            for stage in ["Beginner", "Intermediate", "Advanced"]:
                st.markdown(f"### {stage}")
                for item in roadmap.get(stage, []):
                    st.markdown(f"- {item}")

            st.markdown("### Recommended Certifications")
            for cert in roadmap.get("Certifications", []):
                st.markdown(f"- {cert}")

            st.markdown("### Portfolio Project Ideas")
            for project in roadmap.get("Projects", []):
                st.markdown(f"- {project}")

            st.markdown("### Interview Preparation Tips")
            for tip in roadmap.get("Interview", []):
                st.markdown(f"- {tip}")

            render_feedback_section(predicted_role)

    st.markdown("---")
    st.markdown("### Startup AI Talent Dashboard")
    st.markdown(
        "This experience is designed to look professional, modern, and actionable for early career and reskilling users."
    )


if __name__ == "__main__":
    main()
