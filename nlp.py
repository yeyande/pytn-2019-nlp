from lxml import html
import requests
from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib
import nltk
from nltk.probability import FreqDist
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import wikipedia, random
from gensim import corpora, models
import gensim
matplotlib.use('TkAgg')
[nltk.download(lib) for lib in
    ['punkt',
     'wordnet',
     'averaged_perceptron_tagger',
     'stopwords',
     'vader_lexicon']]

sample_data = ["I'm currently sitting in a classroom at PyTennessee in Nashville learning about Natural Language Processing and how to use it"]

def get_conference_data():
    page = requests.get('https://www.pytennessee.org')
    tree = html.fromstring(page.content)
    conference_data = tree.xpath('/html/body/div[4]/div/div/div[2]/p[string(.)]/text()')
    static_conference_data = ["PyTennessee is a yearly regional Python conference held every February in Nashville, TN. PyTennessee 2019 will be our 6th year, and like every year before it, promises to be our best yet! There's a bit of something for everyone at PyTennessee 2019, and if you'd like to know more about it, please visit our ", '.', 'PyTennessee is a nonprofit event facilitated by TechFed Nashville, a Tennessee nonprofit organization whose mission is to support and grow the grassroots tech talent in Middle Tennessee through educational events and groups. TechFed provides financial stewardship, risk management and volunteer leadership development for Nashville-area technology events.', 'PyTennessee is dedicated to a harassment-free conference experience for everyone. Our anti-harassment policy can be found at: ']
    return conference_data

def get_sentence_tokens(data):
    sentences = [sent_tokenize(sentence) for sentence in data]
    print(sentences)
    return sanitize_special_characters(sentences[0], remove_digits=True)

def sanitize_special_characters(sentences, remove_digits=False):
    pattern = r'/[^\w-]|_/' if not remove_digits else r'[^a-zA-Z\s]'
    return [ re.sub(pattern, '', sentence) for sentence in sentences ]
    

def get_word_tokens(sentences):
    words = [word_tokenize(sentence) for sentence in sentences]
    print(words)
    return [word for word_group in words for word in word_group]

def plot_freq_dist(words, num_words = 20):
    distribution = FreqDist(words)
    distribution.plot(num_words, cumulative=False)

def get_stems(words):
    # this is kind of bad because it fucks up a lot
    ps = PorterStemmer()
    stems = [ps.stem(word) for word in words]
    print(stems)
    return stems

def get_lemmas(words):
    # lemmatizing is a better stemmer
    lemmatizer = WordNetLemmatizer()
    lemma = [lemmatizer.lemmatize(word) for word in words]
    print(lemma)
    return lemma

def get_pos_tags(words):
    tags = [nltk.pos_tag([word]) for word in words]
    print(tags)
    return tags

def get_bag_of_words(sentences):
    vectorizer = CountVectorizer()
    print(vectorizer.fit_transform(sentences).todense())
    print(vectorizer.vocabulary_)

conference_data = get_conference_data()
sentences = get_sentence_tokens(conference_data)
words = get_word_tokens(sentences)
stems = get_stems(words)
lemmas = get_lemmas(words)
sample_sentences = get_sentence_tokens(sample_data)
sample_words = get_word_tokens(sample_sentences)
sample_tags = get_pos_tags(sample_words)
get_bag_of_words(sentences)
get_bag_of_words(sample_sentences)
plot_freq_dist(lemmas, num_words=30)


### Topic Modeling

def fetch_data():
    topics = wikipedia.random(2)
    topics.append('Music')
    topics.append('Gardening')
    topics.append('Reading')
    topics.append('Nashville')
    topics.append('Tennessee')
    print(topics)
    articles = [ [topic, wikipedia.page(topic).content] for topic in topics ]
    return articles

def clean_article(article):
    topic, document = article
    tokens = RegexpTokenizer(r'\w+').tokenize(document.lower())
    clean_tokens = [token
                    for token in tokens
                    if token not in stopwords.words('english')]
    stemmed_tokens = [PorterStemmer().stem(token) for token in clean_tokens]
    return (topic, stemmed_tokens)
    

cleaned_articles = list(map(clean_article, fetch_data()))
wiki_contents = [article[1] for article in cleaned_articles]
dictionary = corpora.Dictionary(wiki_contents)

corpus = [dictionary.doc2bow(article) for article in wiki_contents[:-1]]
lda_model = gensim.models.ldamodel.LdaModel(
                corpus, num_topics=5, id2word = dictionary, passes=100)
print(lda_model.print_topics(num_topics=7, num_words=5))
print(list(lda_model[[dictionary.doc2bow(wiki_contents[-1])]]))


### Sentiment Analysis
hotel_rev = [
"Great place to be when you are in Nashville.", "the place was being renovated when I visited so the seating was limited.", "Loved the ambience, loved the food", "The food is delicious but not amazing.", "Service - little slow, probably because too many people.",]
sid = SentimentIntensityAnalyzer()
for sentence in hotel_rev:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in ss:
        print('{0}: {1}, '.format(k, ss[k]), end='')
    print()
