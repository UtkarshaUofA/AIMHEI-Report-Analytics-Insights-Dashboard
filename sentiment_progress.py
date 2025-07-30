import os
from transformers import pipeline
import matplotlib.pyplot as plt

# Load all transcript files 
def load_transcripts(data_folder):
    transcripts = {}
    for fname in sorted(os.listdir(data_folder)):
        if fname.lower().endswith(".txt"):
            with open(os.path.join(data_folder, fname), 'r', encoding='utf-8') as f:
                transcripts[fname] = f.read()
    return transcripts

# Extract only student lines 
def extract_student_turns(text):
    return [line[3:].strip() for line in text.splitlines() if line.startswith("D:")]

# Load BERT GoEmotions model 
def load_empathy_classifier():
    print("Loading BERT empathy classifier...")
    return pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion", return_all_scores=True)

# Score empathy for each student turn 
def compute_empathy_score(student_turns, classifier):
    empathy_labels = {"caring", "compassion", "empathy", "love", "gratitude", "sadness", "relief"}

    scores = []

    for turn in student_turns:
        results = classifier(turn)[0]  # List of {label, score}
        empathy_score = sum(r["score"] for r in results if r["label"] in empathy_labels)
        scores.append(empathy_score)

    return round(sum(scores) / len(scores), 3) if scores else 0.0

# Loop through each transcript and score 
def process_transcripts(transcripts, classifier):
    empathy_scores = {}

    for i, (fname, content) in enumerate(transcripts.items(), 1):
        student_turns = extract_student_turns(content)
        score = compute_empathy_score(student_turns, classifier)
        session_name = f"Session {i}"
        empathy_scores[session_name] = score
        print(f"{session_name} - {fname}: Empathy Score = {score}")

    return empathy_scores

# Plot trend of empathy scores 
def plot_empathy_trend(scores):
    sessions = list(scores.keys())
    values = [score * 100 for score in scores.values()] 
    # values = list(scores.values())

    plt.figure(figsize=(8, 5))
    plt.plot(sessions, values, marker='o', color='darkgreen')
    plt.title("Empathy Score Over Time")
    plt.xlabel("Session")
    plt.ylabel("Empathy Score (0 to 1)")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main
if __name__ == "__main__":
    data_folder = "data"
    transcripts = load_transcripts(data_folder)
    classifier = load_empathy_classifier()
    scores = process_transcripts(transcripts, classifier)
    plot_empathy_trend(scores)
