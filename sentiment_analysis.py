import pandas as pd
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease, syllable_count,  lexicon_count
nltk.download('vader_lexicon')
# Directory where your text files are stored
directory = (r'C:\Users\namit\OneDrive\Desktop\blackCoffer\new attempt\extracted_articles')

# Initialize a list to store the data for each text file
data = []

sia = SentimentIntensityAnalyzer()

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):  # Check if the file is a text file
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            text_content = file.read()
            
            sentiment_scores = sia.polarity_scores(text_content)
            positive_score = sentiment_scores['pos']
            negative_score = sentiment_scores['neg']
            polarity_score = sentiment_scores['compound']

            reading_ease = flesch_reading_ease(text_content)

            word_count = lexicon_count(text_content, removepunct=True)
            sentence_count = text_content.count('.') + text_content.count('!') + text_content.count('?')
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            # complex_word_percentage = complex_word_count(text_content) / word_count * 100 if word_count > 0 else 0
            # fog_index = 0.4 * (avg_sentence_length + complex_word_percentage)
            avg_word_length = sum(len(word) for word in text_content.split()) / word_count if word_count > 0 else 0
            syllable_per_word = syllable_count(text_content) / word_count if word_count > 0 else 0
            personal_pronouns = sum(1 for word in nltk.word_tokenize(text_content) if word.lower() in ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'])
            subjectivity_score = sentiment_scores['neu'] + sentiment_scores['pos'] + sentiment_scores['neg']
            
            # For now, let's print the filename and content to verify
            data.append({
                'URL_ID': filename[:-4],  # Remove the ".txt" extension
                'URL': '',  # Replace this with the URL if available
                'POSITIVE SCORE': positive_score,
                'NEGATIVE SCORE': negative_score,
                'POLARITY SCORE': polarity_score,
                'SUBJECTIVITY SCORE': subjectivity_score,
                'AVG SENTENCE LENGTH': avg_sentence_length,               
                'AVG NUMBER OF WORDS PER SENTENCE': avg_sentence_length,
                'WORD COUNT': word_count,
                'SYLLABLE PER WORD': syllable_per_word,
                'PERSONAL PRONOUNS': personal_pronouns,
                'AVG WORD LENGTH': avg_word_length
            })
# Once you're done with the analysis, you can proceed to the next steps
df = pd.DataFrame(data)

# Define the order of columns as per your specified structure
columns_order = ['URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE',
                 'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 
                 'AVG NUMBER OF WORDS PER SENTENCE',  'WORD COUNT',
                 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH']

# Reorder columns
df = df[columns_order]
csv = df.to_csv('csv.csv')