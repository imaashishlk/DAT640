import json
import pickle
import pandas as pd
import re
from typing import List, Tuple, Union
from numpy import ndarray
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
import ipytest
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

def load_dataset_as_list(filename):
    """Loads the datasets from a JSON file."""

    """Args:
        path: Path to file from which to load data

    Returns:
        List of questions  and  list of coresponding categories .
    """
    output_data=[]
    dict = {}
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
        for question in data:
            if not question['question']:  # Ignoring null questions
                continue

            dict = {"question":question['question'],
            "category": question['category']}
            output_data.append(dict)
    df = pd.DataFrame.from_dict(output_data)
    question=list(df['question'])
    category = list(df['category'])

    return question, category

def preprocess(doc: str) -> str:
    """Preprocesses questions to prepare it for feature extraction.

    Args:
        doc: String comprising the unprocessed contents of some questions.

    Returns:
        String comprising the corresponding preprocessed text.
    """
    # Remove HTML markup using a regular expression.
    re_html = re.compile("<[^>]+>")
    text = re_html.sub(" ", doc)
    # Replace punctuation marks (including hyphens) with spaces.
    for c in '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~':#string punctuation:
        text = text.replace(c," ")
    # Lowercase and split on whitespaces.
    text1= text.lower().split()
    stop_words = set(['a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in',
     'into', 'is', 'it', 'no', 'not', 'of', 'on', 'or', 'such', 'that', 'the', 'their', 'then', 'there',
      'these', 'they', 'this', 'to', 'was', 'will', 'with'])

    tokens_wo_stopwords= [word for word in text1 if word not in stop_words]
    return " ".join(tokens_wo_stopwords)
    
def preprocess_multiple(docs: List[str]) -> List[str]:
    """Preprocesses multiple texts to prepare them for feature extraction.

    Args:
        docs: List of strings, each consisting of the unprocessed contents
            of some email file.

    Returns:
        List of strings, each comprising the corresponding preprocessed
            text.
    """
    preprocessed_docs = [preprocess(text) for text in docs]
    

    return preprocessed_docs


def extract_features(
    train_dataset: List[str], test_dataset: List[str], tfdif=True
) -> Union[Tuple[ndarray, ndarray], Tuple[List[float], List[float]]]:
    """
    Extracts feature vectors from a preprocessed train and test datasets.

    Args:
        train_dataset: List of strings, each consisting of given question.
        test_dataset: List of strings, each consisting of given question

    Returns:
        A tuple of of two lists. The lists contain extracted features for 
          training and testing dataset respectively.
    """
    if tfdif:
        tfidf_vect= TfidfVectorizer()
        X_train_counts = tfidf_vect.fit_transform(train_dataset)
        X_test_counts = tfidf_vect.transform(test_dataset) 
    else:
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(train_dataset)
        X_test_counts = count_vect.transform(test_dataset) 
    return X_train_counts,X_test_counts


def train_classifier(X: ndarray, y: List[int],sgd=True) -> object:
    """
    Implement the function that takes data produced by feature extraction to train a classifier. 

    The function takes two numerical array-like objects 
    (representing the questions and their categories) as input and returns a trained model object. 
    The model object must have a `predict` method that takes as input a numerical array-like object
     (representing instances) and returns a numerical array-like object (representing predicted labels).
    
    
    Trains a classifier on extracted feature vectors.

    Args:
        X: Numerical array-like object (2D) representing the questions.
        y: Numerical array-like object (1D) representing the categories.

    Returns:
        A trained model object capable of predicting over unseen sets of
            questions.
    """
    if sgd:
        classifier = SGDClassifier()
        classifier_fit= classifier.fit(X,y)
    else:
        classifier = MultinomialNB(alpha=1.0)
        classifier_fit=classifier.fit(X,y)
    return classifier_fit


if __name__ == "__main__":
    print("Loading data..")
    train_questions, train_category = load_dataset_as_list('files_to_process/smarttask_dbpedia_train.json')
    test_questions, test_category = load_dataset_as_list('files_to_process/smarttask_dbpedia_test.json')

    print('Preprocessing questions....')
    preprocces_train_questions= preprocess_multiple(train_questions)
    preprocces_test_questions= preprocess_multiple(test_questions)
    
    print("Extracting features using CountVectorizer...")
    train_feature_count, test_feature_count, = extract_features(
        preprocces_train_questions, preprocces_test_questions,tfdif=False)

    print(20*'-')

    print("Training SVM classifier based on CountVectorizer")
    svm_classifier_count = train_classifier(train_feature_count, train_category,sgd=True)

    print("Applying model on test data...")
    predicted_category_svm_count = svm_classifier_count.predict(test_feature_count)

    print("Evaluating ")
    accuracy_svm_count = metrics.accuracy_score(test_category, predicted_category_svm_count)

    print(f"Accuracy:\t{accuracy_svm_count:.3f}")

    print(20*'-')

    print(f"Saving Predictions (Category).....")

    with open('category_prediction/categories.txt', 'w') as f:
        for item in predicted_category_svm_count:
            f.write(item + '\n')
    

    print(f"Predictions saved: category_prediction/categories.txt")
    
