import csv
import random

OUTPUT_PATH = "career_dataset.csv"
ROWS = 5000

classes = [
    "AI Engineer",
    "Cloud Engineer",
    "Cybersecurity Analyst",
    "Data Analyst",
    "Data Scientist",
    "DevOps Engineer",
    "Mobile App Developer",
    "Non-Tech",
    "Software Engineer",
    "UI/UX Designer",
    "Web Developer",
    "Backend Developer",
    "Frontend Developer",
]

# Core feature prototypes for each career with distinct means.
prototypes = {
    "AI Engineer": {
        "Coding_and_Algorithms": 5,
        "UI_and_Visual_Design": 2,
        "Data_and_Analytics": 4,
        "Math_and_Predictive_Modeling": 5,
        "Infrastructure_and_Automation": 3,
        "Security_and_Networking": 2,
        "Business_and_Product_Strategy": 2,
        "System_Architecture_and_APIs": 4,
    },
    "Cloud Engineer": {
        "Coding_and_Algorithms": 4,
        "UI_and_Visual_Design": 2,
        "Data_and_Analytics": 3,
        "Math_and_Predictive_Modeling": 3,
        "Infrastructure_and_Automation": 5,
        "Security_and_Networking": 3,
        "Business_and_Product_Strategy": 2,
        "System_Architecture_and_APIs": 4,
    },
    "Cybersecurity Analyst": {
        "Coding_and_Algorithms": 3,
        "UI_and_Visual_Design": 1,
        "Data_and_Analytics": 3,
        "Math_and_Predictive_Modeling": 3,
        "Infrastructure_and_Automation": 4,
        "Security_and_Networking": 5,
        "Business_and_Product_Strategy": 2,
        "System_Architecture_and_APIs": 3,
    },
    "Data Analyst": {
        "Coding_and_Algorithms": 3,
        "UI_and_Visual_Design": 3,
        "Data_and_Analytics": 5,
        "Math_and_Predictive_Modeling": 2,
        "Infrastructure_and_Automation": 2,
        "Security_and_Networking": 2,
        "Business_and_Product_Strategy": 5,
        "System_Architecture_and_APIs": 2,
    },
    "Data Scientist": {
        "Coding_and_Algorithms": 4,
        "UI_and_Visual_Design": 2,
        "Data_and_Analytics": 4,
        "Math_and_Predictive_Modeling": 5,
        "Infrastructure_and_Automation": 3,
        "Security_and_Networking": 2,
        "Business_and_Product_Strategy": 3,
        "System_Architecture_and_APIs": 3,
    },
    "DevOps Engineer": {
        "Coding_and_Algorithms": 4,
        "UI_and_Visual_Design": 1,
        "Data_and_Analytics": 2,
        "Math_and_Predictive_Modeling": 3,
        "Infrastructure_and_Automation": 5,
        "Security_and_Networking": 4,
        "Business_and_Product_Strategy": 2,
        "System_Architecture_and_APIs": 4,
    },
    "Mobile App Developer": {
        "Coding_and_Algorithms": 5,
        "UI_and_Visual_Design": 4,
        "Data_and_Analytics": 3,
        "Math_and_Predictive_Modeling": 2,
        "Infrastructure_and_Automation": 3,
        "Security_and_Networking": 2,
        "Business_and_Product_Strategy": 3,
        "System_Architecture_and_APIs": 4,
    },
    "Non-Tech": {
        "Coding_and_Algorithms": 1,
        "UI_and_Visual_Design": 2,
        "Data_and_Analytics": 2,
        "Math_and_Predictive_Modeling": 1,
        "Infrastructure_and_Automation": 1,
        "Security_and_Networking": 1,
        "Business_and_Product_Strategy": 5,
        "System_Architecture_and_APIs": 1,
    },
    "Software Engineer": {
        "Coding_and_Algorithms": 5,
        "UI_and_Visual_Design": 2,
        "Data_and_Analytics": 3,
        "Math_and_Predictive_Modeling": 3,
        "Infrastructure_and_Automation": 4,
        "Security_and_Networking": 3,
        "Business_and_Product_Strategy": 2,
        "System_Architecture_and_APIs": 4,
    },
    "UI/UX Designer": {
        "Coding_and_Algorithms": 2,
        "UI_and_Visual_Design": 5,
        "Data_and_Analytics": 2,
        "Math_and_Predictive_Modeling": 1,
        "Infrastructure_and_Automation": 1,
        "Security_and_Networking": 1,
        "Business_and_Product_Strategy": 4,
        "System_Architecture_and_APIs": 1,
    },
    "Web Developer": {
        "Coding_and_Algorithms": 4,
        "UI_and_Visual_Design": 4,
        "Data_and_Analytics": 3,
        "Math_and_Predictive_Modeling": 2,
        "Infrastructure_and_Automation": 3,
        "Security_and_Networking": 2,
        "Business_and_Product_Strategy": 3,
        "System_Architecture_and_APIs": 4,
    },
    "Backend Developer": {
        "Coding_and_Algorithms": 5,
        "UI_and_Visual_Design": 2,
        "Data_and_Analytics": 3,
        "Math_and_Predictive_Modeling": 3,
        "Infrastructure_and_Automation": 4,
        "Security_and_Networking": 3,
        "Business_and_Product_Strategy": 2,
        "System_Architecture_and_APIs": 5,
    },
    "Frontend Developer": {
        "Coding_and_Algorithms": 4,
        "UI_and_Visual_Design": 5,
        "Data_and_Analytics": 2,
        "Math_and_Predictive_Modeling": 2,
        "Infrastructure_and_Automation": 3,
        "Security_and_Networking": 2,
        "Business_and_Product_Strategy": 3,
        "System_Architecture_and_APIs": 3,
    },
}

feature_names = [
    "Coding_and_Algorithms",
    "UI_and_Visual_Design",
    "Data_and_Analytics",
    "Math_and_Predictive_Modeling",
    "Infrastructure_and_Automation",
    "Security_and_Networking",
    "Business_and_Product_Strategy",
    "System_Architecture_and_APIs",
]

# Balanced counts across classes
base = ROWS // len(classes)
remainder = ROWS % len(classes)
counts = {c: base for c in classes}
for idx, c in enumerate(classes[:remainder]):
    counts[c] += 1

# Sampling noise patterns
def sample_value(mean):
    value = int(round(random.gauss(mean, 0.6)))
    return max(1, min(5, value))

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=feature_names + ["Career Label"])
    writer.writeheader()
    for career in classes:
        for _ in range(counts[career]):
            row = {feat: sample_value(prototypes[career][feat]) for feat in feature_names}
            row["Career Label"] = career
            writer.writerow(row)

print(f"Generated {OUTPUT_PATH} with {sum(counts.values())} rows and {len(classes)} classes.")