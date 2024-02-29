
with open('/Users/fabiocat/Documents/git/cmi/voice_type_classifier/output_voice_type_classifier/5272290_MRI_recording/chi.rttm', 'r') as f:
    text_data = f.read()

# Splitting the provided text data into lines for processing
lines = text_data.strip().split("\n")

# Parsing each line to extract start time, duration, and label, then calculate end time and format the output
with open('label2.txt', 'w') as f:
    for line in lines:
        parts = line.split()  # Splitting each line by spaces
        start = float(parts[3])
        duration = float(parts[4])
        label = parts[7]
        end = start + duration  # Calculating the end time
        f.write(f"{start}\t{end}\t{label}\n")
