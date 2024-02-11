import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class TextPreprocessor:

    def __init__(self):
        '''
        Initialize the TextPreprocessor.

        Initializes the set of English stopwords using NLTK.
        '''
        self.stopwords_set = set(stopwords.words('english'))

    def preprocess(self, text):
        '''
        Preprocess the input text.

        Parameters:
        - text (str): Input text to be preprocessed.

        Returns:
        - str: Preprocessed text.
        '''
        # Convert to lowercase
        text = text.lower()
        # Remove non-alphanumeric characters
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenize and remove stopwords
        text = ' '.join([word for word in word_tokenize(text) if word not in self.stopwords_set])
        return text
