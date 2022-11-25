import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tflearn.data_utils import VocabularyProcessor
from sklearn.model_selection import train_test_split
import tensorflow
import json


enron_data_path = "../dataset/enron"
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()




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

def feature_extraction_bagofwords(emails_list):
    # currently do not consider max_features
    tfidfv = TfidfVectorizer(
        decode_error = "ignore",
        analyzer = "word",
        stop_words = "english",
        smooth_idf = False,
        max_features = 35000,# at most 2010 features for two datasets
    )
    x = np.array(emails_list)
    x = tfidfv.fit_transform(x)
    x = x.toarray()
    vocab_path = "../output/vocabulary_bagofwords.txt"
    print("output the vocabulary to " + vocab_path + " ......\n")
    # with open(vocab_path, 'w') as f:
    #     f.write(json.dumps(tfidfv.vocabulary_))
    print("len of x: ", len(x))
    print("#features: ",len(x[0]))
    return x



def feature_extraction_vo(emails_list):
    vp = VocabularyProcessor(
        max_document_length = 1000,
        min_frequency = 1,
        vocabulary = None,
        tokenizer_fn = None
    )
    x = vp.fit_transform(emails_list)
    x = np.array(list(x))
    print(x)
    vocab_path = "../output/vocabulary_tf.txt"
    # with open(vocab_path, 'w') as f:
    #     f.write(json.dumps(vp.vocabulary_._mapping))
    print("len of x: ", len(x))
    return x


def main():
    print("\n\n")
    spam_list, ham_list = process_all_data()
    emails_list = spam_list + ham_list
    len_spam = len(spam_list)
    len_ham = len(ham_list)
    y = [1] * len_spam + [0] * len_ham
    x = feature_extraction_bagofwords(emails_list)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

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
