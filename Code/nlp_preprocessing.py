### ---- Data preprocessing, frequency analysis, and Word Cloud ---- ###


# import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import string
import nltk
from nltk import word_tokenize


# read the file
df = pd.read_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv')


# clean review date
df['reviews.date'] = pd.to_datetime(df['reviews.date'])
print("The dateset dates from: ",df['reviews.date'].min()," to ", df['reviews.date'].max())


# frequency tables
print("Primary categories:")
print(df.primaryCategories.value_counts())


print("Product names:")
print(df.name.value_counts())


# create new binary label class: sentiment (postive = 1, negative = 0)
df['sentiment'] = df['reviews.rating'].apply(lambda x: '1' if x > 3 else '0')
print(df['sentiment'].value_counts())


# plots on target variables
# sentiment
ax1 = sns.countplot(x=df['sentiment'], data=df)
for p in ax1.patches:
   ax1.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
plt.show()

# review ratings
ax2 = sns.countplot(x = 'reviews.rating', data=df)
for p in ax2.patches:
   ax2.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
plt.show()

# recommendation
ax3 = sns.countplot(x = 'reviews.doRecommend', data=df)
for p in ax3.patches:
   ax3.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
plt.show()


# ---- text-preprocessing -----
# lower-case and remove punctuations
punct = set(string.punctuation)
def remove_punct(text):
     clean = [w.lower() for w in word_tokenize(text) if w not in punct]
     return clean

df['punct'] = df['reviews.text'].apply(lambda x: remove_punct(x))

# lemmatization
def lemmatize(token): # text: any sentence (untokenized)
     wnl = nltk.WordNetLemmatizer()
     clean = [wnl.lemmatize(w) for w in token]
     return clean

#df['lemma'] = df.punct.apply(lambda x: lemmatize(x))

# remove stopwords
stop_words = nltk.corpus.stopwords.words('english')

def remove_stopword(text):
     clean = [w for w in text if w not in stop_words]
     return clean

# remove custom stopwords that contains product names
stopwords = ['amazon', 'tablet','kindle','echo','alexa','fire','kids','\'s']

def remove_stopword_custom(text):
     clean = [w for w in text if w not in stopwords]
     return clean

df.punct = df.punct.apply(lambda x: remove_stopword(x))

df['clean_tokens'] = df.punct.apply(lambda x: remove_stopword_custom(x))

df['clean_text'] = df.clean_tokens.apply(lambda x: " ".join(x))



# select columns for analysis
df_short = df[['clean_text','clean_tokens','reviews.doRecommend','reviews.rating','sentiment']]
print(df_short.head())

df_short.to_csv('df_short.csv')


# filter out positive vs. negative reviews based on rating
df_pos = df_short[df['reviews.rating']>3]
df_neg = df_short[df['reviews.rating']<3]


pos_tokens =list(df_pos['clean_tokens'])
neg_tokens =list(df_neg['clean_tokens'])

# create list of tokens for pos vs. neg
pos_lst = []
for s in pos_tokens:
    pos_lst.extend(s)
    
neg_lst = []
for s in neg_tokens:
    neg_lst.extend(s)
    
print(len(pos_lst))
print(len(neg_lst))

# top words in positive reviews
cnt_pos = Counter(pos_lst)
top_pos = cnt_pos.most_common(30)
print(top_pos)

# top words in negative reviews
cnt_neg = Counter(neg_lst)
top_neg = cnt_neg.most_common(30)
print(top_neg)


top_pos_lst = [w[0] for w in top_pos]
top_neg_lst = [w[0] for w in top_neg]


pos_cnt_all = Counter(top_pos_lst)
neg_cnt_all = Counter(top_neg_lst)


# word cloud for positive reviews
wordcloud_pos = WordCloud(width=600, height=400,
                      background_color='white',
                      min_font_size=10).generate_from_frequencies(pos_cnt_all)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud_pos)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()


# word cloud for negative reviews
wordcloud_pos = WordCloud(width=600, height=400,
                      background_color='white',
                      stopwords=STOPWORDS,
                      min_font_size=10).generate_from_frequencies(neg_cnt_all)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud_pos)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()
