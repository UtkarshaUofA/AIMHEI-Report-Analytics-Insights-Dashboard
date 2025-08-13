import os
from transformers import pipeline

# Load BERT-based NER model
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

data_folder = "Data"
output_folder = "cleaned_data"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(data_folder):
    if filename.lower().endswith(".txt"):
        file_path = os.path.join(data_folder, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            transcript_text = f.read()

        cleaned_lines = []
        for line in transcript_text.splitlines():
            # Run NER on the full line (Doctor or Patient)
            entities = ner_pipeline(line)
            new_line = line
            # Replace from end to start so indices don't shift
            for ent in reversed(entities):
                if ent["entity_group"] == "PER":  # 'PER' = person name
                    start, end = ent["start"], ent["end"]
                    new_line = new_line[:start] + "[NAME]" + new_line[end:]
            cleaned_lines.append(new_line)

        cleaned_text = "\n".join(cleaned_lines)

        output_path = os.path.join(output_folder, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        print(f"Processed {filename} → {output_path}")

print("All transcripts processed.")
