import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tflearn.data_utils import VocabularyProcessor

import tensorflow
import json
import warnings
warnings.filterwarnings('ignore')

enron_data_path = "../dataset/enron"



# return the list of preprocess email text
def preprocess_one_set(data_path):
    res = []
    for root, dirs, files in os.walk(data_path):
        for file_name in files:
            cur_file_path = os.path.join(root, file_name)
            cur_txt = ""
            with open(cur_file_path) as cur_f:
                try:
                    cur_lines = cur_f.readlines()
                    for one_line in cur_lines:
                        one_line = one_line.strip('\n')
                        one_line = one_line.strip('\r')
                        cur_txt += one_line
                    res.append(cur_txt)
                except UnicodeDecodeError:
                    continue
    return res

def process_all_data():
    spam_list = []
    ham_list = []
    NUM_SET = 1
    for i in range(1, NUM_SET + 1):
        spam_path = enron_data_path + str(i) +  "/spam/"
        ham_path = enron_data_path + str(i) + "/ham/"
        cur_spam_list = preprocess_one_set(spam_path)
        cur_ham_list = preprocess_one_set(ham_path)
        spam_list += cur_spam_list
        ham_list += cur_ham_list
    print("len of spam_list: ", len(spam_list))
    print("len of ham_list: ", len(ham_list))
    print("\n\n")

    return spam_list, ham_list

def feature_extraction_wordbags():
    spam_list, ham_list = process_all_data()
    emails_list = spam_list + ham_list
    
    # currently do not consider max_features
    tfidfv = TfidfVectorizer(
        decode_error = "ignore",
        analyzer = "word",
        stop_words = "english",
        smooth_idf = False,
        #max_features = 2010,# at most 2010 features for two datasets
    )
    x = np.array(emails_list)
    cnt = tfidfv.fit_transform(x)
    cnt = cnt.toarray()
    vocab_path = "../output/vocabulary_wordbags.txt"
    print("output the vocabulary to " + vocab_path + " ......\n")
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(tfidfv.vocabulary_))
    print("len of x: ", len(cnt))
    print("#features: ",len(cnt[0]))



def feature_extraction_vo():
    spam_list, ham_list = process_all_data()
    emails_list = spam_list + ham_list
    vp = VocabularyProcessor(
        max_document_length = 100,
        min_frequency = 1,
        vocabulary = None,
        tokenizer_fn = None
    )
    x = vp.fit_transform(emails_list)
    x = np.array(list(x))
    print(x)
    vocab_path = "../output/vocabulary_tf.txt"
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(vp.vocabulary_._mapping))

    print("len of x: ", len(x))


def main():
    print("\n\n")
    feature_extraction_wordbags()
    #feature_extraction_vo()



if __name__ == "__main__":
    main()

#-----------some results--------------
# len of spam_list:  1487
# len of ham_list:  3672
# 5159
# #features:  73103

# len of spam_list:  2855
# len of ham_list:  8032
# 10887
# #features:  2010
