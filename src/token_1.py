import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



enron_data_path = "../dataset/enron1/"

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
    spam_path = enron_data_path + "spam/"
    ham_path = enron_data_path + "ham/"
    spam_list = preprocess_one_set(spam_path)
    ham_list = preprocess_one_set(ham_path)
    print(len(spam_list))
    print(spam_list[1])
    print()
    print(len(ham_list))
    print(ham_list[1])

def main():
    process_all_data()

if __name__ == "__main__":
    main()