import json
from my_utils.semantic_similarity import SemanticSimilarityCalculator
import re

def get_protocol_instructions(instructions_json):
    """
    Get the list of instructions from the instructions json file.
    """
    with open(instructions_json, 'r') as f:
        data = json.load(f)
        instructions = []
        for ppt in data:
            text = ""
            for segment in ppt['segments']:
                text += segment['text']
            instructions.append(text.strip())
    return instructions

"""
def get_sessions(sessions_json):
    with open(sessions_json, 'r') as f:
        data = json.load(f)
        sessions = []
        for child in data:
            session = []
            for segment in child['segments']:
                if 'words' in segment:
                    for word in segment['words']:
                        if 'start' in word and 'end' in word and 'word' in word:
                            #print(segment)
                            start = word['start']
                            #print(speaker)
                            end = word['end']
                            text = word['word']
                            #print(text)
                            #print(old_text)
                            session.append([text, start, end])
                else:
                    print("NO WORDS")
            sessions.append(session)
    return sessions
"""

def get_sessions(sessions_json):
    with open(sessions_json, 'r') as f:
        data = json.load(f)
        sessions = []
        for child in data:
            session = []
            for segment in child['segments']:
                if 'start' in segment and 'end' in segment and 'text' in segment:
                    #print(segment)
                    start = segment['start']
                    #print(speaker)
                    end = segment['end']
                    text = segment['text']
                    #print(text)
                    #print(old_text)
                    session.append([text.strip(), start, end])
                else:
                    print("NO WORDS")
            sessions.append(session)
    return sessions

def calculate_similarity(sentence, instruction, similarity_calculator):
    """
    Calculate semantic similarity between sentences and a list of instructions.
    Returns mean similarities for each sentence.
    """
    sentence_embedding = similarity_calculator.extract_semantic_embeddings(sentence)
    phrase_embedding = similarity_calculator.extract_semantic_embeddings(instruction)
    similarity = similarity_calculator.compute_cosine_similarity(sentence_embedding, phrase_embedding)
    return similarity

similarity_calculator = SemanticSimilarityCalculator()

instructions_json = '../data/instructions/speech_language_instructions.json'
session_json = '../data/5253316_speech_language_transcription 2.json'
#session_json = '../data/iktae_test.json'
#session_json = '../data/5030023_speech_language_non_transcription.json'
actual_instructions = []
instructions = get_protocol_instructions(instructions_json)
entire_session = get_sessions(session_json)



#print("instructions")
#print(instructions)
#print("entire_session")
#print(entire_session)
#print(len(entire_session[0]))

    #print(sentence_start)



script = entire_session[0]
stop = False
audacity = ""
try:
    while len(script) > 0 and not stop:
        first_sentence, first_sentence_start, first_sentence_end = script[0]
        """
        print("first_sentence")
        print(first_sentence)
        print("first_sentence_start")
        print(first_sentence_start)
        print("first_sentence_end")
        print(first_sentence_end)
        """
        for i, task in enumerate(instructions):
            print("task")
            print(task)

            actual_instruction = ""
            actual_instruction_start = 0
            actual_instruction_end = 0

            for instruct in re.split('[.?!]', task):
                #print("instruct")
                #print(instruct)
                if instruct == "" or (instruct.count(' ') <= 2 and instruct not in str(script)):
                    continue
                if actual_instruction != "":
                    last_sentence_actual_instruction = re.split("[.?!]", actual_instruction)
                    #print("last_sentence_actual_instruction")
                    #print(last_sentence_actual_instruction)
                    last_sentence_actual_instruction = last_sentence_actual_instruction[-2]
                    similarity = calculate_similarity(last_sentence_actual_instruction, instruct, similarity_calculator)
                    #print("similarity")
                    #print(similarity)
                    if similarity > 0.9:
                        continue
                
                while actual_instruction == "" and len(script) > 0:
                    similarity = calculate_similarity(first_sentence, instruct, similarity_calculator)
                    #print("similarity")
                    #print(similarity)
                    if similarity < 0.9:
                        script = script[1:]
                        first_sentence, first_sentence_start, first_sentence_end = script[0]
                        """
                        print("first_sentence")
                        print(first_sentence)
                        print("first_sentence_start")
                        print(first_sentence_start)
                        print("first_sentence_end")
                        print(first_sentence_end)
                        """
                    else:
                        break

                similarity = calculate_similarity(first_sentence, instruct, similarity_calculator)
                #print("similarity")
                #print(similarity)
                if similarity > 0.9:
                    if actual_instruction == "":
                        actual_instruction_start = first_sentence_start
                    actual_instruction_end = first_sentence_end
                    # if instruct == re.split("[.!?]", task)[-1]:
                    #    actual_instruction_end = first_sentence_end
                    actual_instruction += first_sentence
                #else:
                #    actual_instruction_end = first_sentence_start
                
                script = script[1:]
                first_sentence, first_sentence_start, first_sentence_end = script[0]
                """
                print("first_sentence")
                print(first_sentence)
                print("first_sentence_start")
                print(first_sentence_start)
                print("first_sentence_end")
                print(first_sentence_end)
                """
            audacity += f"{actual_instruction_start}\t{actual_instruction_end}\t{i}\n"
            print("actual_instruction")
            print(actual_instruction)
            stop = True

except Exception as e:
    print(e)
    print(script)
    print("instruct")
    print(instruct)

print(audacity)
with open('../data/audacity.txt', 'w') as f:
    f.write(audacity)




"""
segmented_session = []
for task in instructions:
    actual_instruction = ""
    actual_instruction_start = 0
    actual_instruction_end = 0

    for i, instruct in enumerate(task.split(".")):
        print("instruct")
        print(instruct)
        print(i)
        print(len(task.split(".")))
        
        for sentence_text, sentence_start, sentence_end in entire_session[0]:
            print("sentence_text")
            print(sentence_text)

            similarity = calculate_similarity(sentence_text, instruct, similarity_calculator)
            print("similarity")
            print(similarity)
            if similarity > 0.95:
                if actual_instruction == "":
                    actual_instruction_start = sentence_start
                actual_instruction += sentence_text
                print("actual_instruction_start")
                print(actual_instruction_start)
                print("actual_instruction")
                print(actual_instruction)
                #print(instruction)
            else:
                actual_instruction_end = sentence_start
                break
                
        
    print("actual_instruction_end")
    print(actual_instruction)
    input("aab")
    print("\n")
"""

