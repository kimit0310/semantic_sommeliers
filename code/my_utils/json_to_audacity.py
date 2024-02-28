import json

# Load data from JSON file
with open('/Users/iktae.kim/Desktop/semantic_sommeliers/data/iktae_test.json', 'r') as f:
    data = json.load(f)

# Write data to TXT file
with open('label.txt', 'w') as f:
    for file in data:
        for segment in file.get('segments', []):
            start = segment.get('start', '')
            end = segment.get('end', '')
            speaker = segment.get('speaker', '')
            f.write(f'{start}\t{end}\t{speaker}\n')