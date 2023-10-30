# filter some words? stops
import pke
import spacy
import nltk
from nltk.stem.porter import PorterStemmer
from summarizer import Summarizer

summarizer_model = Summarizer()

nlp = spacy.load("en_core_web_sm")

#we can also filter out specific stopwords
# import string
#all_stopwords = string.punctuation
#from nltk.corpus import stopwords
#all_stopwords = all_stopwords + stopwords.words('english')


# performs stemming with nltk
# inputs:
# text: String
# remove_stopwords: bool
# stopwords: list of stopwords
# outputs:
# list of stemmed tokens
def stem_tokenize(text, remove_stopwords=False, stopwords=None):
  stemmer = PorterStemmer()
  tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
  if remove_stopwords:
      tokens = [word for word in tokens if not(word in stopwords)]
  return [stemmer.stem(word) for word in tokens]

# performs lemmatization with SpaCy
# inputs:
# text: String
# outputs:
# list of lemmas
def lemmatize(text):
    doc = nlp(text)
    doc_lemmas = []
    for token in doc:
        lemma = token.lemma_
        doc_lemmas.append(lemma)
    return doc_lemmas

# inputs:
# text: String
# token_type: lemmas|stems|tokens
# Topk: int (max number of most frequent words)
# outputs:
# token2count: dictionary with the mapping between the tokens and their counts
# most_frequent_tokens: list of most frequent tokens
def get_counts(text, token_type, topk=5):
    if token_type=='lemmas':
        tokens = lemmatize(text)
    elif token_type=='stems':
        tokens = stem_tokenize(text)
    else:
        tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    token2count = dict()
    for t in tokens:
        if not(t in token2count):
            token2count[t] = 0
        token2count[t]+=1
    count2tokens = dict()
    for k,v in token2count.items():
        if not v in count2tokens:
            count2tokens[v] = []
        count2tokens[v].append(k)
    all_counts = sorted(list(count2tokens.keys()), reverse=True)[:topk]
    most_frequent_tokens = []
    for count in all_counts:
        most_frequent_tokens.extend(count2tokens[count])
    return token2count, most_frequent_tokens[:topk]

# retrieves Topk sentences for the abstractive summarization
# inputs:
# text: String
# Topk: int (number of summary sentences)
def get_summary(text, topk=3):
    return summarizer_model(text, num_sentences=topk)

# retrieves Topk keywords
# inputs:
# text: String
# Topk: int (max number of keywords)
# outputs:
# keywords: list of keywords
def get_keywords(text, topk=5):
    out=[]
    kw_extractor = pke.unsupervised.MultipartiteRank()
    kw_extractor.load_document(text)
    # extract only content words with the following POS tags:
    allowed_pos = {'PROPN', 'VERB', 'ADJ', 'NOUN'}
    kw_extractor.candidate_selection(pos=allowed_pos)
    kw_extractor.candidate_weighting()
    #print(kw_extractor.get_n_best(n=Topk))
    keywords = [kw[0] for kw in kw_extractor.get_n_best(n=topk)]
    
    return keywords
    

### code execution example ###
text = "Cake is a flour confection made from flour, sugar, and other ingredients, and is usually baked. In their oldest forms, cakes were modifications of bread, but cakes now cover a wide range of preparations that can be simple or elaborate, and which share features with desserts such as pastries, meringues, custards, and pies. The most common ingredients include flour, sugar, eggs, fat (such as butter, oil or margarine), a liquid, and a leavening agent, such as baking soda or baking powder. Common additional ingredients include dried, candied, or fresh fruit, nuts, cocoa, and extracts such as vanilla, with numerous substitutions for the primary ingredients. Cakes can also be filled with fruit preserves, nuts or dessert sauces (like custard, jelly, cooked fruit, whipped cream or syrups),[1] iced with buttercream or other icings, and decorated with marzipan, piped borders, or candied fruit. Cake is often served as a celebratory dish on ceremonial occasions, such as weddings, anniversaries, and birthdays. There are countless cake recipes; some are bread-like, some are rich and elaborate, and many are centuries old. Cake making is no longer a complicated procedure; while at one time considerable labor went into cake making (particularly the whisking of egg foams), baking equipment and directions have been simplified so that even the most amateur of cooks may bake a cake."    
token_type = 'tokens' # 'tokens' 'lemmas' 'stems'
topk = 5
token2count, most_frequent_tokens = get_counts(text, token_type, topk)
print('token2count:', token2count)
print('most_frequen_tokens:', most_frequent_tokens)
print('summary:', get_summary(text))
print('keywords:', get_keywords(text))



