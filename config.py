import os

class Config:
    ############## PARAMS ##############
    data_folder = "data"
    instructions_folder = os.path.join(data_folder, "instructions")
    sessions_folder = "data/sessions"
    cross_correlations_folder = os.path.join(data_folder, "cross_correlation")
    story1 = 'It was the first day of school. It was a tough day for all the kids. One girl had a really hard time because nobody could say her name. Her name was Peggy Babcock. Go ahead. Try and say it three times quickly. “Peggy Babcock Peggy Babcock Peggy Babcock.” Not easy going, right? She was afraid to say hello to any of the other kids on the playground. One boy walked up to her and asked what her name was. She said “When you hear my name it sounds simple but no one can say it. It is Peggy Babcock.” He laughed and said “Your name is tricky but mine is better. It sounds simple but no one can remember it. It is Jonas Norvin Sven Arthur Schwinn Bart Winston Ulysses M.” Peggy laughed and said “Easy. Your name sounds like Joan is nervous when others win. But you win some, you lose some. How do you like my version?" Jonas was so happy that he said “Lets be friends. I will call you PB.” The pair of them stuck so close to each other that everyone at school called them “PB and J.”'
    story2 = 'Some time ago, in a place neither near nor far, there lived a king who did not know how to count, not even to zero. Some say this is the reason he would always wish for more — more food, more gold, more land. He simply did not realize how much he already owned. Everyone in his kingdom could do the math and tally bushels of corn, loaves of bread, and urns of gold. But how would they measure the height of his castle or the stretch of his kingdom? You might think “Aaah, ooh, easy — just measure it in meters!” But in those days, the useless unit of measure was based on stains splattered along the king s cloak while drinking shrub juice. The kingdom needed a new way of counting distance. “A kingdom without a proper ruler,” proclaimed the king, “is like riches without measure.” He launched a challenge amid trumpets, drums, flags and cannons. “The person who creates a unit of measure fit for a ruler will be rewarded beyond measure!” A tall order indeed! The first person to come forward was a bulky locksmith with a stiff jaw. He approached the king with an air of secrecy and whispered, “I have the key to measure the kingdom, but only I can wield it.” He then rubbed his beard and pulled the key from his locks of oily hair. The key turned out to be a hair itself! “Judge the reach of my vast kingdom with a hair s width?” laughed the king. “What a poor idea. That would take forever or longer!” The second person eager for the prize was a fidgety boy who knew all numbers (including zero). He produced a curious object from one of his many pockets. It was a complex shape that seemed to change proportions depending on which direction you gazed upon it. The boy said in a measured voice, “This polyhedron has many edges, with each edge of a different length. Only a king could be counted on to use it justly.” He gave the king an awful earful of an explanation that went on and on. The long and the short of it was that the king could make no more use of it than of a puddle of spilled oatmeal. Finally, a little girl with a big idea tugged on the mismeasured cloak of the king. The king sized up the little girl with the big idea and said “I don’t have time for this, and for that matter, I have no concept of space, either.” The girl looked up, then down, then spun around and blurted out: “Aren’t you able to solve the puzzle yourself? Why must you break up your kingdom into tiny pieces when everything around you is Humpty Dumpty together again? Your kingdom IS a unit and you are the ruler.” The king — startled, befuddled, and bemused — found the words wise. He aimed to be satisfied with all around him, big or small or somewhere in between.'
    stories = [story1, story2]
    ############## PARAMS ##############
    normalization = True
    filtering = True
    new_sample_rate = 16000
    lowcut = 500.0
    highcut = 7500.0
    seconds_threshold = 3.0
    story_absolute_peak_height = 0.65
    long_instructions_peak_height = 0.6
    word_instructions_peak_height = 0.8
    non_word_instructions_peak_height = 0.8
    long_instructions_absolute_peak_height = 0.01
    word_instructions_absolute_peak_height = 0.05
    non_word_instructions_absolute_peak_height = 0.05
    ############## PARAMS ##############