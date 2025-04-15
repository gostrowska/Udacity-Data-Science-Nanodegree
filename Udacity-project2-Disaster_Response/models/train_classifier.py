import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import re
import numpy as np
import pandas as pd
import pickle

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn import multioutput
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    INPUT:
    database_filepath - path to database
    
    OUTPUT:
    X - messages 
    Y - categories of the messages

    """
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages_categories', engine) 
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = df.columns[4:]
    
    return X, Y, category_names


def tokenize(text):
    
    """
    INPUT:
    text - raw text
    
    OUTPUT:
    clean_tokens - tokenized messages
    """
    
    #normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
        
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
     
    #remove stop words
    clean_tokens = [w for w in clean_tokens if w not in stopwords.words('english')]

    return clean_tokens


def build_model():
    """
    OUTPUT:
    model = ML model pipeline after parameter tuning with grid search
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {
    #         'vect__ngram_range': ((1, 1), (1, 2)),
    #         'clf__estimator__n_estimators': [50, 100, 200],
    #         'clf__estimator__min_samples_split': [2, 3, 4],
#             'clf__estimator__n_estimators': [10, 20, 40],
            'clf__estimator__learning_rate': [0.01, 0.05]
#             'clf__estimator__min_samples_split': [2, 3]
    }

    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model


def evaluate_model(model, X_test, y_test, category_names):
    """
    INPUT:
    model - ML model
    X_test - test messages
    y_test - categories for test messages
    category_names - category name for y from load data function
    """
    
    y_pred = model.predict(X_test)

    print(classification_report(y_test.values, y_pred, target_names=category_names))
    


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()