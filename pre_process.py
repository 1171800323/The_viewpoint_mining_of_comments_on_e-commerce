import numpy as np
import pandas as pd


def read_review(datatype="Train"):
    if datatype == "Train":
        file = datatype + "/Train_reviews.csv"
    elif datatype == "Test":
        file = datatype + "/Test_reviews.csv"
    return pd.read_csv(file, sep=",")


def read_label(datatype="Train"):
    file = datatype + "/Train_labels.csv"
    return pd.read_csv(file, sep=",")


def read_task(task='task1'):
    if task == 'task1':
        file = 'Test/task1_answer.csv'
    else:
        file = 'Test/task2_answer.csv'
    asp_opi = []
    with open(file, 'r', encoding='utf-8') as f:
        rows = f.read().strip().split('\n')
    for row in rows:
        words = row.split(',')
        print(words)
        asp = words[1]
        opi = words[2]
        result = ''
        if asp != '_':
            result += asp
        if opi != '_':
            result += opi
        asp_opi.append(result)
    return rows, asp_opi


def pre_process(review, label, max_len=69):
    token_label = np.zeros((review.shape[0], max_len))
    for rowid, row in label.iterrows():
        idx = row["id"] - 1
        if row["A_start"] != " " and row["A_end"] != " ":
            start = int(row["A_start"])
            end = int(row["A_end"])
            if end - start == 1:
                token_label[idx, start] = 1  # B-ASP
            else:
                token_label[idx, start] = 1  # B-ASP
                token_label[idx, start + 1: end] = 2  # I-ASP
        if row["O_start"] != " " and row["O_end"] != " ":
            start = int(row["O_start"])
            end = int(row["O_end"])
            if end - start == 1:
                token_label[idx, start] = 3  # B-OPI
            else:
                token_label[idx, start] = 3  # B-OPI
                token_label[idx, start + 1: end] = 4  # I-OPI
    train_data = []
    for rowid, row in review.iterrows():
        idx = row["id"] - 1
        string = row["Reviews"]
        term_id = token_label[idx]
        for idx, ch in enumerate(string):
            seq = term_id[idx]
            if ch != ' ':
                if seq == 1:
                    train_data.append(ch + ' B-ASP')
                elif seq == 2:
                    train_data.append(ch + ' I-ASP')
                elif seq == 3:
                    train_data.append(ch + ' B-OPI')
                elif seq == 4:
                    train_data.append(ch + ' I-OPI')
                else:
                    train_data.append(ch + ' O')

        train_data.append('\n')
    with open('Train/data.data', 'w', encoding='utf-8') as f:
        for word in train_data:
            if word != '\n':
                f.write(word + '\n')
            else:
                f.write(word)


def get_asp_opi_combined_labels():
    label = read_label()
    asp_opi = []
    category_labels = []
    polarity_labels = []
    word_list = []
    for rowid, row in label.iterrows():
        category_labels.append(row['Categories'])
        polarity_labels.append(row['Polarities'])
        asp = row['AspectTerms']
        opi = row['OpinionTerms']
        combined = ''
        if asp != '_':
            combined += asp
            word_list.append(asp)
        if opi != '_':
            combined += opi
            word_list.append(opi)
        asp_opi.append(combined)
    return asp_opi, category_labels, polarity_labels, word_list


if __name__ == "__main__":
    # df_review = read_review()
    # df_label = read_label()
    # pre_process(df_review, df_label)
    # a, b, c, d = get_asp_opi_combined_labels()
    # df_review = read_review('Test')
    # df_review = df_review.sort_values(by='id', ascending=True)
    # df_review.to_csv('test.csv', sep=',', header=True, index=False)
    read_task()
