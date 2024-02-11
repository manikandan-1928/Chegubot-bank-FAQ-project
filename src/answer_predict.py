from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.utils import *

class Chatbot:

    def __init__(self, data_embed, qn_cleaning, tokenizer, bert_model):
        '''
        Initialize the Chatbot with required data and models.

        Parameters:
        - data_embed (pd.DataFrame): DataFrame containing embedded data.
        - qn_cleaning (TextPreprocessor): TextPreprocessor object for question cleaning.
        - tokenizer (BertTokenizer): BERT tokenizer.
        - bert_model (BertModel): Pre-trained BERT model.
        '''
        self.data_embed = data_embed
        self.category = {i + 1: j for i, j in enumerate(data_embed['Class'].unique().tolist())}
        self.qn_cleaning = qn_cleaning
        self.tokenizer = tokenizer
        self.bert_model = bert_model

    def preprocess_and_embed_user_query(self, user_query):
        '''
        Preprocess and embed the user's query using BERT.

        Parameters:
        - user_query (str): User's input query.

        Returns:
        - Tuple[str, np.ndarray]: Cleaned query and embedded representation.
        '''
        cleaned_query = self.qn_cleaning.preprocess(user_query)
        tokenized_query = self.tokenizer(cleaned_query, return_tensors='pt', padding=True, truncation=True)
        embedded_query = self.bert_model(**tokenized_query)['last_hidden_state'][:, 0, :].detach().numpy()
        return cleaned_query, embedded_query

    def calculate_cosine_similarities(self, embedded_data, embedded_usr):
        '''
        Calculate cosine similarities between user's query and pre-embedded data.

        Parameters:
        - embedded_data (List[np.ndarray]): List of embedded data.
        - embedded_usr (np.ndarray): Embedded representation of user's query.

        Returns:
        - List[float]: List of cosine similarities.
        '''
        cos_sims = []
        for embedded_query in embedded_data:
            sims = cosine_similarity(np.array(embedded_query).reshape(1, -1), embedded_usr.reshape(1, -1))
            cos_sims.append(sims[0][0])
        return cos_sims

    def predict_answer(self, user_query, selected_class, filtered_data, class_indices):
        '''
        Predict the answer based on user's query and selected class.

        Parameters:
        - user_query (str): User's input query.
        - selected_class (str): Selected class for prediction.
        - filtered_data (pd.DataFrame): DataFrame containing filtered data for the selected class.
        - class_indices (dict): Dictionary containing class indices.

        Returns:
        - Tuple[str, str, List[str], List[str]]: Predicted question, predicted answer, top 5 similar questions, and top 5 answers.
        '''
        class_start_index = class_indices[selected_class]
        cleaned_query, embedded_usr = self.preprocess_and_embed_user_query(user_query)

        cos_sims = self.calculate_cosine_similarities(filtered_data['Embedded_Query'], embedded_usr)

        ind = cos_sims.index(max(cos_sims))
        index = class_start_index + ind

        top_5_qns = []
        top_5_ans = []

        inds = get_max5(cos_sims)

        for i in inds:
            top_5_qns.append(filtered_data['Question'].loc[class_start_index + i])
            top_5_ans.append(filtered_data['Answer'].loc[class_start_index + i])

        predicted_question = filtered_data['Question'].loc[index]
        predicted_answer = filtered_data['Answer'].loc[index]

        print(top_5_qns)
        print(top_5_ans)

        return predicted_question, predicted_answer, top_5_qns, top_5_ans
