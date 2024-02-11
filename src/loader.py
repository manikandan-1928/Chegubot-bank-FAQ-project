from transformers import BertTokenizer, BertModel
import pandas as pd
import ast

def load_data(file_path):
    '''
    Load data from a CSV file with embedded queries.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded DataFrame.
    '''
    data = pd.read_csv(file_path, converters={'Embedded_Query': ast.literal_eval})
    return data

def load_tokenizer(tokenizer_path):
    '''
    Load a BERT tokenizer from a specified path.

    Parameters:
    - tokenizer_path (str): Path to the tokenizer.

    Returns:
    - BertTokenizer: Loaded BERT tokenizer.
    '''
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def load_bert_model(model_path):
    '''
    Load a pre-trained BERT model from a specified path.

    Parameters:
    - model_path (str): Path to the BERT model.

    Returns:
    - BertModel: Loaded BERT model.
    '''
    bert_model = BertModel.from_pretrained(model_path)
    return bert_model

def load_all_data(tokenizer_path, model_path, data_path):
    '''
    Load all required data for the chatbot.

    Parameters:
    - tokenizer_path (str): Path to the BERT tokenizer.
    - model_path (str): Path to the BERT model.
    - data_path (str): Path to the CSV file containing embedded data.

    Returns:
    - pd.DataFrame: Loaded DataFrame containing embedded data.
    - BertTokenizer: Loaded BERT tokenizer.
    - BertModel: Loaded BERT model.
    '''
    data = load_data(data_path)
    tokenizer = load_tokenizer(tokenizer_path)
    bert_model = load_bert_model(model_path)
    return data, tokenizer, bert_model
