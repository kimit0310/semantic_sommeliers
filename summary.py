import os
import glob
import csv


def summarize_instructions(audacity_folder):
    # This dictionary will store the session summary
    session_summaries = []

    # List all text files in the audacity directory
    audacity_files = glob.glob(os.path.join(audacity_folder, "*.txt"))

    # Define all possible audio instructions (00 to 37)
    total_audio_instructions = [f"{i:02d}" for i in range(38)]
    total_stories = ["story_0", "story_1"]

    # Iterate over each file
    for file_path in audacity_files:
        session_name = os.path.basename(file_path).replace(".txt", "")
        detected_audio = set()
        detected_stories = set()

        with open(file_path, "r") as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                label = parts[2]

                if label in total_stories:
                    detected_stories.add(label)
                elif "_" in label:
                    audio_index = label.split("_")[0]
                    detected_audio.add(audio_index)

        # Determine missing audio instructions and stories
        missed_audio = sorted(
            set(total_audio_instructions) - detected_audio, key=lambda x: int(x)
        )
        missed_stories = sorted(set(total_stories) - detected_stories)

        # Store results in the dictionary
        session_summaries.append(
            {
                "Session": session_name,
                "Total Audio Instructions Detected": len(detected_audio),
                "Total Stories Detected": len(detected_stories),
                "Missed Audio Instructions": ", ".join(missed_audio),
                "Missed Stories": ", ".join(missed_stories),
            }
        )

    # Output the summary to a CSV file
    csv_file_path = os.path.join(audacity_folder, "summary_report.csv")
    with open(csv_file_path, "w", newline="") as csvfile:
        fieldnames = [
            "Session",
            "Total Audio Instructions Detected",
            "Total Stories Detected",
            "Missed Audio Instructions",
            "Missed Stories",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for summary in session_summaries:
            writer.writerow(summary)

    print(f"Summary report saved to {csv_file_path}")


# Change folder here for now

if __name__ == "__main__":
    audacity_folder = (
        "/Users/iktae.kim/semantic_sommeliers/data/data_run_20240425_131233/audacity"
    )
    summarize_instructions(audacity_folder)
