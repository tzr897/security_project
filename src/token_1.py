import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



enron_data_path = "../dataset/enron1/"

e1 = []
for root, dirs, files in os.walk(enron_data_path):
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
                e1.append(cur_txt)
            except UnicodeDecodeError:
                continue

for i in range(0,10):
    print(type(e1[i]))
