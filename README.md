# Sequence_Labelling
The goal of this assignment is to get some experience with sequence labeling. We will be assigning dialogue acts to sequences of utterances in conversations from a corpus.

In sequence labeling it is often beneficial to optimize the tags assigned to the sequence as a whole rather than treating each tag decision separately

Training data : <a href = "https://github.com/Niranjani29/Sequence_Labelling/blob/master/train.zip"></a>

Corpus tools file :
Provides three functions and two data containers:
get_utterances_from_file - loads utterances from an open csv file
get_utterances_from_filename - loads utterances from a filename
get_data - loads all the CSVs in a directory
DialogUtterance - A namedtuple with various utterance attributes
PosTag - A namedtuple breaking down a token/pos pair

Feel free to import, edit, copy, and/or rename to use in your assignment.

Baseline Tagger :

Step 1 :
Add training data to the Trainer object using the append method which takes two arguments (feature_vector_list,label_list) and loads the training data for a single sequence. In our case, each sequence corresponds to a dialogue, and the feature_vector_list is a list of feature vectors (one for each utterance in the dialogue). The label_list corresponds to the dialogue acts for those utterances. Each feature vector is a list of individual features which are binary. The presence of a feature indicates that it is true for this item. Absence indicates that the feature would be false. Here are the features for a training example using features for whether a particular token is present or not in an utterance.
['TOKEN_i', 'TOKEN_certainly', 'TOKEN_do', 'TOKEN_.']

Step 2: 

<a href = "https://github.com/Niranjani29/Sequence_Labelling/blob/master/baseline_tagger.py"></a>
In the baseline feature set, for each utterance you include:
• a feature for whether or not the speaker has changed in comparison with the previous utterance.
• a feature marking the first utterance of the dialogue.
• a feature for every token in the utterance (see the description of CRFsuite for an example).
• a feature for every part of speech tag in the utterance (e.g., POS_PRP POS_RB POS_VBP POS_.).

ADVANCED :

<a href = "https://github.com/Niranjani29/Sequence_Labelling/blob/master/advanced_tagger.py"></a>
I have worked on the punctuations. I have made two changes 
	(1) If the Utterance is a question, than I have appended a string "QUESTION" like "FIRST_UTTERANCE" and "NO_WORDS"
	(2) I personally feel that commas dont play a vital role in this assignment. So I have eliminated tokens that are commas(",")

Step 3:
Print the output labels to 'output.txt'  <a href = "https://github.com/Niranjani29/Sequence_Labelling/blob/master/output.txt"></a>
