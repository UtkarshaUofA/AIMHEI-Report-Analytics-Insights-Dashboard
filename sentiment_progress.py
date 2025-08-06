import os
import re
from transformers import pipeline
import matplotlib.pyplot as plt

# Load all transcript files 
def load_transcripts(data_folder):
    transcripts = {}
    for fname in sorted(os.listdir(data_folder)):
        if fname.lower().endswith(".txt"):
            with open(os.path.join(data_folder, fname), 'r', encoding='utf-8') as f:
                content = f.read()
                transcripts[fname] = content
                print(f"Loaded {fname} | Length: {len(content)}")
    return transcripts

# Extract only student lines (e.g., "D: ...")

def extract_doctor_lines(transcript: str) -> list:
    # Use regex to match lines starting with a number, then D:, then the actual dialogue
    pattern = r'^\d+:\s*D:\s*(.+)$'
    matches = re.findall(pattern, transcript, re.MULTILINE)
    return matches

# Load BERT GoEmotions model 
def load_empathy_classifier():
    print("Loading BERT empathy classifier...")
    try:
        return pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion", top_k=None)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# # Labels considered empathetic
# goemotions_labels = [
#     "admiration", "amusement", "anger", "annoyance", "approval",
#     "caring", "confusion", "curiosity", "desire", "disappointment",
#     "disapproval", "disgust", "embarrassment", "excitement", "fear",
#     "gratitude", "grief", "joy", "love", "nervousness", "optimism",
#     "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
# ]

# Score empathy for each student turn 
def compute_empathy_score(student_turns, classifier):
    empathy_labels = {"caring", "love", "gratitude", "sadness", "relief", "grief", "joy"}  # Adjusted to model's actual labels
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
        student_turns = extract_doctor_lines(content)
        print(f"\n{fname} - Student Turns Found: {len(student_turns)}")
        for turn in student_turns:
            print(f"  > {turn}")

        score = compute_empathy_score(student_turns, classifier)
        session_name = f"Session {i}"
        empathy_scores[session_name] = score
        print(f"{session_name} - {fname}: Empathy Score = {score}")

    return empathy_scores

# Plot trend of empathy scores 
import matplotlib.pyplot as plt
import os

def plot_empathy_scores(scores_dict, output_folder="empathy_graphs", filename="empathy_score_plot.png"):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    sessions = list(scores_dict.keys())
    scores = list(scores_dict.values())

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(sessions, scores, marker='o', linestyle='-', linewidth=2)

    # Titles and labels
    plt.title("Empathy Scores Over Time", fontsize=16)
    plt.xlabel("Session", fontsize=12)
    plt.ylabel("Empathy Score", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Rotate x-labels for readability
    plt.xticks(rotation=45)

    # Save and show
    output_path = os.path.join(output_folder, filename)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✅ Graph saved at: {output_path}")

    
# Main
if __name__ == "__main__":
    data_folder = "data"
    transcripts = load_transcripts(data_folder)
    classifier = load_empathy_classifier()

    if classifier:
        scores = process_transcripts(transcripts, classifier)
        print(f"\nFinal Empathy Scores: {scores}")
        plot_empathy_scores(scores)
    else:
        print("Classifier not loaded. Exiting.")

