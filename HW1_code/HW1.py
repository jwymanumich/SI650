from collections import defaultdict
import math
import matplotlib.pyplot as plt
import nltk
from nltk.stem import *
import operator
import plotly.express as px
import plotly.graph_objects as go
import re

def remove_stopwords_from_collection(collection):
    ''' Return a new collection with no stop words
    '''

    collection_out = {}
    collection_out['Name'] = collection['Name']
    collection_out['documents'] = []

    for document in collection['documents']:
        out_document = {'line':document['line'], "words":[], 'POS':[], "STOP_WORD":[]}

        for index, sw in enumerate(document['STOP_WORD']):
            if(sw == False):
                out_document['words'].append(document['words'][index])
                out_document['POS'].append(document['POS'][index])
        collection_out['documents'].append(out_document)

    return collection_out

def remove_stopwords_from_inverted_index(inverted_index):
    ''' Return a new inverted index with no stop words
    '''

    out_inverted_index = {}
    for cur_word in inverted_index:
        if(inverted_index[cur_word]['STOP_WORD'] == "False"):
            out_inverted_index[cur_word] = inverted_index[cur_word]
    return out_inverted_index

def frequency_of_stopwords(inverted_index):
    '''percentage of the word occurrences that are stopwords.
    counted from inverted_index and multipled by occurence count'''

    words_total = sum(inverted_index[item]['total_frequency'] for item in inverted_index)
    stop_words_total = sum(inverted_index[item]['total_frequency'] for item in inverted_index if inverted_index[item]['STOP_WORD'] is "True")
    return [stop_words_total, words_total, float(stop_words_total)/words_total]

def percentage_of_capital_letters(collection):
    ''' Count the percentage of total characters that are upper case.
        This needs to use the collection to insure that we are not losing
        case infromation in the inverted_indes
    '''

    upper_case_count = 0
    lower_case_count = 0

    for row in collection['documents']:
        for c in row['line']:
            if (c.islower() is True):
                lower_case_count += 1
            elif (c.isupper() is True):
                upper_case_count += 1

    return [upper_case_count, lower_case_count, float(upper_case_count)/(upper_case_count + lower_case_count)]

def average_number_of_characters_per_word(inverted_index):
    ''' Calculate the average number of characters per word
        We can do this faster with the inverted_index and 
        multiplying value by total_frequency
    '''

    total_chars = 0
    total_words = 0

    for item in inverted_index:
        inverted_index_item = inverted_index[item]
        total_words += inverted_index_item['total_frequency']
        total_chars += len(item) * inverted_index_item['total_frequency']

    return [total_chars, total_words, float(total_chars)/total_words]

def percentage_of_nouns_adjectives_verbs_adverbs_pronouns(collection):
    ''' Count from collection to maintain contextual information in each location.
    '''

    noun_count = 0
    adj_count = 0
    verb_count = 0
    adv_count = 0
    pronoun_count = 0
    total_words = 0
    
    for document in collection['documents']:
        total_words += len(document['words'])
        for pos in document['POS']:
            if(pos.startswith("N")):
                noun_count += 1
            elif(pos.startswith("J")):
                adj_count += 1
            elif(pos.startswith("V")):
                verb_count += 1
            elif(pos.startswith("RB")):
                adv_count += 1
            elif(pos.startswith("PR") or pos.startswith("WP")):
                pronoun_count += 1

    return {"Noun": [noun_count, total_words, float(noun_count)/total_words],
    "Adjective": [adj_count, total_words, float(adj_count)/total_words],
    "Verb": [verb_count, total_words, float(verb_count)/total_words],
    "Adverb": [adv_count, total_words, float(adv_count)/total_words],
    "Pronoun": [pronoun_count, total_words, float(pronoun_count)/total_words]}


def top_nouns_verbs_adjectives(collection):
    ''' Count most frequent occurences of noun, verb, adj
        Use noun to maintain contextual information.
    '''

    counter = {'N':defaultdict(lambda: 0), "V":defaultdict(lambda: 0), "J":defaultdict(lambda: 0)}
    for document in collection['documents']:
        for index, word in enumerate(document['words']):
            pos = document['POS'][index]
            if(pos.startswith("N")):
                counter['N'][word.lower()] += 1
            elif(pos.startswith("V")):
                counter['V'][word.lower()] += 1
            elif(pos.startswith("J")):
                counter['J'][word.lower()] += 1

    response = {}
    for pos in [["N", "Noun"], ["V", "Verb"], ["J", "Adjectives"]]:
        sorted_words = sorted(counter[pos[0]].items(), key=lambda k_v: k_v[1], reverse=True) #2010
        response[pos[1]] = sorted_words[0:10]

    return response


def tfidf(collection, inverse_index):
    ''' 
    '''

    collection_tfidf = []
    total_documents = len(collection['documents'])

    for document in collection['documents'][0:10]:
        document_tf = defaultdict(lambda: 0)
        document_tfidf = {}

        # get a list of document words and occurences
        for index, word in enumerate(document['words']):
            document_tf[word] += 1

        # For each document word
        for word in document_tf:

            # Calculate the TF value
            # T F (t, d) = log(c(t, d) + 1)
            tf_value = math.log(1 + document_tf[word])

            # IDF(t) = 1 + log(N/k).
            document_frequency = len(set(inverse_index[word]['doc_ids']))
            idf = 1 + math.log(total_documents/document_frequency)
            
            # put TF and IDF for 'word' together 
            # and store by document
            document_tfidf[word] = tf_value*idf

        # Store document tfidf information by collection
        collection_tfidf.append(document_tfidf)
    return collection_tfidf


def plot_data(name, inverted_index):
    ''' Plat the data to a graph and a log log graph
    '''

    x = []
    y = []

    # Count the number of words with each number of occurences
    value_count = defaultdict(lambda: 0)
    for key in inverted_index:
        value_count[int(inverted_index[key]['total_frequency'])] += 1

    vocabulary = len(inverted_index)

    #Use unique words
    for k in sorted(value_count.items(), key=lambda k_v: k_v[0], reverse=True):
        y.append(float(k[1])/vocabulary)
        x.append(k[0])

    plt.plot(x, y, "ro")
    plt.ylabel('occurences')
    plt.xlabel('rank order')
    plt.title(name)
    plt.show()

    plt.plot(x, y, "ro")
    plt.ylabel('log occurences')
    plt.xlabel('log rank order')
    plt.xscale('log') 
    plt.yscale('log') 
    plt.title(name)
    plt.show()

def load_stop_words():
    ''' Load words from the file, and strip carraige returns'''

    s = set()
    for line in open('stoplist.txt', "r", encoding="utf-8").readlines():
        s.add(line.strip('\n'))
    return s

def load_data(file_name, stop_words):
    ''' Load words from file, skipping items matching values
    in the provided set of stop_words'''

    stemmer = PorterStemmer()

    inverted_index = defaultdict(lambda: {'total_frequency' : 0, "POS":'', "STOP_WORD":'False', "doc_ids": [], "frequency":[]})
    my_collection = {"Name":file_name, 'documents':[]}

    cur_record = 1
    for line in open(file_name, "r", encoding="utf-8").readlines():
        document = {'line':line, "words":[], 'POS':[], "STOP_WORD":[]}
        line_tok = nltk.word_tokenize(line)
        for word_pos in nltk.pos_tag(line_tok):

            cur_word = word_pos[0].lower()
            s = stemmer.stem(cur_word)

            x = re.search("[a-zA-Z]", s)

            if(x is not None):
                document['words'].append(cur_word)

                inverted_index_item = inverted_index[cur_word]
                inverted_index_item['total_frequency'] += 1

                inverted_index_item['POS'] = word_pos[1]
                document['POS'].append(word_pos[1])
                document['STOP_WORD'].append(cur_word in stop_words)

                if(cur_record not in inverted_index_item['doc_ids']):
                    inverted_index_item['doc_ids'].append(cur_record)
                    inverted_index_item['frequency'].append(1)
                else:
                    index = inverted_index_item['doc_ids'].index(cur_record)
                    inverted_index_item['frequency'][index] += 1
        my_collection['documents'].append(document)
        cur_record += 1

    for cur_word in inverted_index:
        if(cur_word.lower() in stop_words):
            inverted_index[cur_word]['STOP_WORD'] = "True"

    return inverted_index, my_collection
  
global_stop_words = load_stop_words()

inverted_index_medhelp, collection_medhelp = load_data("medhelp.txt", global_stop_words)
inverted_index_ehr, collection_ehr = load_data("ehr.txt", global_stop_words)

collection_medhelp_no_stop_words = remove_stopwords_from_collection(collection_medhelp)
collection_ehr_no_stop_words = remove_stopwords_from_collection(collection_ehr)

inverted_index_medhelp_no_stop_words = remove_stopwords_from_inverted_index(inverted_index_medhelp)
inverted_index_ehr_no_stop_words = remove_stopwords_from_inverted_index(inverted_index_ehr)

plot_data("medhelp", inverted_index_medhelp_no_stop_words)
plot_data("ehr", inverted_index_ehr_no_stop_words)

print("Q2.2 stats on {} and {}".format(collection_medhelp['Name'], collection_ehr['Name']))
print("Q2.2a - Frequency of Stopwords.")
print("medhelp - {}".format(frequency_of_stopwords(inverted_index_medhelp)[2]))
print("ehr - {}".format(frequency_of_stopwords(inverted_index_ehr)[2]))

print("Q2.2b - Percentage of capital letters")
print("medhelp - {}".format(percentage_of_capital_letters(collection_medhelp)[2]))
print("ehr - {}".format(percentage_of_capital_letters(collection_ehr)[2]))

print("Q2.2c - Average Number of Characters per word")
print("medhelp - {}".format(average_number_of_characters_per_word(inverted_index_medhelp)[2]))
print("ehr - {}".format(average_number_of_characters_per_word(inverted_index_ehr)[2]))

print("Q2.2d - Percentage of Nouns, Adjectives, Verbs, Adverbs, and Pronouns")
r1 = percentage_of_nouns_adjectives_verbs_adverbs_pronouns(collection_medhelp)
r2 = percentage_of_nouns_adjectives_verbs_adverbs_pronouns(collection_ehr)

for key in list(r1):
    print("{}\t{}\t{}".format(key, r1[key][2], r2[key][2]))

print("2.2e - The Top 10 Nouns, Top 10 Verbs, and Top 10 Adjectives.")
r1 = top_nouns_verbs_adjectives(collection_medhelp_no_stop_words)
r2 = top_nouns_verbs_adjectives(collection_ehr_no_stop_words)

for key in list(r1):
    print("\n{}".format(key))
    for item in range(0,10):
        print("medhelp - {}".format(r1[key][item]))
    for item in range(0,10):
        print("ehr - {}".format(r2[key][item]))

print("Q2.3 TF-IDF top scores Medhelp")
for idx, document_tfidf in enumerate(tfidf(collection_medhelp_no_stop_words, inverted_index_medhelp_no_stop_words)):
    print("Document {}".format(idx+1))

    sorted_tfidf = sorted(document_tfidf.items(), key=lambda k_v: k_v[1], reverse=True)

    for tfidf_item in sorted_tfidf[0:5]:
        print("\tword:{} values:{}".format(tfidf_item[0], tfidf_item[1]))


print("Q2.3 TF-IDF top scores Ehr")
for idx, document_tfidf in enumerate(tfidf(collection_ehr_no_stop_words, inverted_index_ehr_no_stop_words)):
    print("Document {}".format(idx+1))

    sorted_tfidf = sorted(document_tfidf.items(), key=lambda k_v: k_v[1], reverse=True)

    for tfidf_item in sorted_tfidf[0:5]:
        print("\tword:{} values:{}".format(tfidf_item[0], tfidf_item[1]))
