# basic python imports are permitted
import sys
import csv
import random

# numpy and pandas are also permitted
import numpy as np
import pandas as pd

def make_bow(data, vocab):
    """
    Produce the bag-of-word representation of the data, along with a vector
    of labels. You *may* use loops to iterate over `data`. However, your code
    should not take more than O(len(data) * len(vocab)) to run.

    Parameters:
        `data`: a list of `(review, label)` pairs, like those produced from
                `list(csv.reader(open("trainvalid.csv")))`
        `vocab`: a list consisting of all unique words in the vocabulary

    Returns:
        `X`: A data matrix of bag-of-word features. This data matrix should be
             a numpy array with shape [len(data), len(vocab)].
             Moreover, `X[i,j] == 1` if the review in `data[i]` contains the
             word `vocab[j]`, and `X[i,j] == 0` otherwise.
        `t`: A numpy array of shape [len(data)], with `t[i] == 1` if
             `data[i]` is a positive review, and `t[i] == 0` otherwise.
    """
    X = np.zeros([len(data), len(vocab)])
    t = np.zeros([len(data)])
    vocab_dict = {}
    for j in range(len(vocab)):
      vocab_dict[vocab[j]] = j
    for i in range(len(data)):
      for word in data[i].split():
        if word in vocab:
          X[i][vocab_dict[word]] = 1
        

    return X

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def predict(x):
    """
    Helper function to make prediction for a given input x.
    This code is here for demonstration purposes only.
    """
    y = []
    ind = np.argmax(x, axis=1)
    for indice in ind:
        y.append(['Dubai', 'New York City', 'Paris', 'Rio de Janeiro'][indice])
    # randomly choose between the four choices: 'Dubai', 'Rio de Janeiro', 'New York City' and 'Paris'.
    # NOTE: make sure to be *very* careful of the spelling/capitalization of the cities!!

    # return the prediction
    return y


def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    # read the file containing the test data
    # you do not need to use the "csv" package like we are using
    # (e.g. you may use numpy, pandas, etc)
    data = pd.read_csv(filename, thousands=',')
    
    data.fillna({'Q1':data['Q1'].median()}, inplace=True)
    data.fillna({'Q2':data['Q2'].median()}, inplace=True)
    data.fillna({'Q3':data['Q3'].median()}, inplace=True)
    data.fillna({'Q4':data['Q4'].median()}, inplace=True)
    data.fillna({'Q7':data['Q7'].mean()}, inplace=True)
    data.fillna({'Q8':data['Q8'].mean()}, inplace=True)
    data.fillna({'Q9':data['Q9'].mean()}, inplace=True)

    data[['Q5_Partner', 'Q5_Friends', 'Q5_Siblings', 'Q5_Co-worker']] = data['Q5'].str.extract(r'(Partner)?,?(Friends)?,?(Siblings)?,?(Co-worker)?')
    data['Q5_Partner'] = data['Q5_Partner'].apply(lambda x: 0 if pd.isna(x) else 1)
    data['Q5_Friends'] = data['Q5_Friends'].apply(lambda x: 0 if pd.isna(x) else 1)
    data['Q5_Siblings'] = data['Q5_Siblings'].apply(lambda x: 0 if pd.isna(x) else 1)
    data['Q5_Co-worker'] = data['Q5_Co-worker'].apply(lambda x: 0 if pd.isna(x) else 1)

    data[['Q6_Skyscraper', 'Q6_Sport', 'Q6_Art_and_Music', 'Q6_Carnival', 'Q6_Cuisine', 'Q6_Economics']] = data['Q6'].str.extract(r'Skyscrapers=>(\d),Sport=>(\d),Art and Music=>(\d),Carnival=>(\d),Cuisine=>(\d),Economic=>(\d)')


    data.dropna(subset=['Q6_Skyscraper'], inplace=True) # lets also drop all empty Q6 since I can't think how to fill those values

    null_mask = data["Q5"].isnull()
    null_rows = data[null_mask]

    data.loc[null_mask, 'Q5_Siblings'] = data['Q5_Siblings'].median()
    data.loc[null_mask, 'Q5_Friends'] = data['Q5_Friends'].median()
    data.loc[null_mask, 'Q5_Partner'] = data['Q5_Partner'].median()
    data.loc[null_mask, 'Q5_Co-worker'] = data['Q5_Co-worker'].median()

    data.drop(columns=['Q5'], inplace=True)
    data.drop(columns=['Q6'], inplace=True)

    data.loc[data['Q7'].gt(45), 'Q7'] = 45
    data.loc[data['Q7'].lt(-20), 'Q7'] = -20
    data.loc[data['Q8'].gt(10), 'Q8'] = 10
    data.loc[data['Q8'].lt(1), 'Q8'] = 1
    data.loc[data['Q9'].gt(15), 'Q9'] = 15
    data.loc[data['Q9'].lt(1), 'Q9'] = 1

    comparison_result1 = (data[['Q6_Sport', 'Q6_Art_and_Music', 'Q6_Carnival', 'Q6_Cuisine', 'Q6_Economics']].eq(data['Q6_Skyscraper'], axis=0)).all(axis=1)
    comparison_result2 = (data[['Q6_Skyscraper', 'Q6_Art_and_Music', 'Q6_Carnival', 'Q6_Cuisine', 'Q6_Economics']].eq(data['Q6_Sport'], axis=0)).all(axis=1)
    comparison_result3 = (data[['Q6_Skyscraper', 'Q6_Sport', 'Q6_Carnival', 'Q6_Cuisine', 'Q6_Economics']].eq(data['Q6_Art_and_Music'], axis=0)).all(axis=1)
    comparison_result4 = (data[['Q6_Skyscraper', 'Q6_Sport', 'Q6_Art_and_Music', 'Q6_Cuisine', 'Q6_Economics']].eq(data['Q6_Carnival'], axis=0)).all(axis=1)
    comparison_result5 = (data[['Q6_Skyscraper', 'Q6_Sport', 'Q6_Art_and_Music', 'Q6_Carnival', 'Q6_Economics']].eq(data['Q6_Cuisine'], axis=0)).all(axis=1)
    comparison_result6 = (data[['Q6_Skyscraper', 'Q6_Sport', 'Q6_Art_and_Music', 'Q6_Carnival', 'Q6_Cuisine']].eq(data['Q6_Economics'], axis=0)).all(axis=1)
    data = data[(~comparison_result1) | (~comparison_result2) | (~comparison_result3) | (~comparison_result4) | (~comparison_result5) | (~comparison_result6)]

    data['Q6_Skyscraper'] = pd.to_numeric(data['Q6_Skyscraper'])
    data['Q6_Art_and_Music'] = pd.to_numeric(data['Q6_Art_and_Music'])
    data['Q6_Carnival'] = pd.to_numeric(data['Q6_Carnival'])
    data['Q6_Cuisine'] = pd.to_numeric(data['Q6_Cuisine'])
    data['Q6_Economics'] = pd.to_numeric(data['Q6_Economics'])

    with open("vocab.txt", 'r') as f:
        vocab = f.readline().split()
    data.dropna(inplace=True)
    data['Q10'] = data['Q10'].str.replace('[^a-zA-Z ]', '', regex=True)
    bow = make_bow(data['Q10'].to_numpy(), vocab)

    data.drop(columns=['Q10'], inplace=True)
    data.drop(columns=['id'], inplace=True)

    data = data.to_numpy()
    data = data.astype(float)
    data = np.hstack((data, bow))

    
    weights = []
    i = 0
    with open("weight.txt", 'r') as f:
        while line := f.readline():
            if line[:-1] == "Start" :
                weight = []
            elif line[:-1] == "End":  
                weights.append(np.array(weight))
                i+=1
            else:
                weight.append([float(x) for x in line[:-1].split()])

    intercepts = []
    with open("intercept.txt", 'r') as f:
        while line := f.readline():
            if line[:-1] == "Start":
                bias = []
            elif line[:-1] == "End":
                intercepts.append(np.array(bias))
            else:
                bias.append(float(line[:-1]))
                
    holder = data
    for y in range(i):
        holder = holder@weights[y] + intercepts[y]
        if y < i - 1:
            holder = np.tanh(holder)
        else:
            holder = softmax(holder)
        # obtain a prediction for this test example
    prediction = predict(holder)
    return prediction

