import nltk
import spacy
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import json




nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")

sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

data = pd.read_csv("training_dataset.csv")

def detect_negative_sentiment(text):
    result = sentiment_model(text)[0]
    is_negative = result['label'] == 'NEGATIVE'
    sentiment_score = result['score']
    return is_negative, sentiment_score

def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    return lemmatized_text

def preprocess_text(sentences):
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    processed_sentences = []
    
    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence.lower())
        tokens = [word for word in tokens if word not in stop_words and len(word) > 1 and not word.isdigit()]
        processed_sentences.append(" ".join(tokens))
    
    return processed_sentences

def vectorize_text(processed_sentences):
    cv = CountVectorizer(ngram_range=(1, 2), binary=True)
    X = cv.fit_transform(processed_sentences)
    return X, cv

def train_model(X, y):
    model = SVC(kernel='linear')
    model.fit(X, y)
    return model

def neutralize_with_gpt(sentence):
    prompt = f"Please neutralize the following biased sentence: {sentence}"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    output = gpt_model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7, do_sample=True)
    neutralized_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
    neutralized_sentence = neutralized_sentence.replace(prompt, "").strip()
    return neutralized_sentence

def analyze_and_modify_bias_in_text(document, model, vectorizer):
    sentences = sent_tokenize(document)
    processed_sentences = preprocess_text(sentences)
    X_new = vectorizer.transform(processed_sentences)
    predictions = model.predict(X_new)

    modified_sentences = []
    
    for i, sentence in enumerate(sentences):
        sentiment, sentiment_score = detect_negative_sentiment(sentence)
        if sentiment and sentiment_score >= 0.75:
            modified_sentence = neutralize_with_gpt(sentence)
            modified_sentences.append(modified_sentence)
        else:
            modified_sentences.append(sentence)

    return modified_sentences

processed_sentences = preprocess_text(data["sentence"])
X, cv = vectorize_text(processed_sentences)
y = data["label"]

model = train_model(X, y)

with open("vocabulary.json", "w") as f:
    json.dump(cv.vocabulary_, f)

document = '''People from certain regions are often stereotyped as being lazy or unmotivated. For example, many people believe that individuals from rural areas are less industrious than those living in cities. This stereotype often leads to discrimination in the job market, as urban workers are seen as more capable and hardworking.
Additionally, there are assumptions about the intelligence of people based on where they were born. Some believe that individuals from less developed countries are less educated, which contributes to a lack of opportunity and inequality. This kind of bias can also affect the way people are treated in social and professional settings, even when they possess the same qualifications and skills as others.

'''

modified_sentences = analyze_and_modify_bias_in_text(document, model, vectorizer=cv)

modified_df = pd.DataFrame({
    'modified_sentence': modified_sentences
})

modified_df.to_csv('modified_dataset.csv', index=False)

print("Modified dataset saved as 'modified_dataset.csv'")
