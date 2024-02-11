# Chatbot Bank FAQ

## Overview

Chatbot Bank FAQ is a project that leverages natural language processing (NLP) techniques to provide users with relevant answers to frequently asked questions related to banking. The project uses a pre-trained BERT model to understand user queries and retrieve the most appropriate answers from a dataset.

## Features

- **Category-Based Querying:** Users can choose from predefined categories (security, loans, accounts, insurance, investments, fundstransfer, cards) to get answers related to specific banking topics.

- **Predictive Answering:** The chatbot predicts answers to user queries by calculating cosine similarities between the input query and pre-embedded data.

- **Top 5 Recommendations:** Users can request the top 5 most relevant questions based on their query and choose an answer from the provided list.

## Project Structure

The project is structured as follows:

- `src/`: Contains source code for the chatbot, data loading, and utility functions.
- `data/`: Stores the dataset and any necessary embeddings.
- `static/`: Includes static files for the web application, such as stylesheets.
- `templates/`: Contains HTML templates for rendering the web interface.
- `app.py`: The main Flask application file.

## Requirements

- Python 3.x
- Flask
- Transformers library
- pandas
- scikit-learn
- nltk

## Getting Started

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/manikandan-1928/Chegubot-bank-FAQ-project.git
   ```

2. **Install Dependencies:**

   ```bash
    pip install -r requirements.txt
   ```

3. **Run the Application:**

    ```bash
    python app.py
   ```


Open your browser and navigate to http://localhost:5000/ to interact with the chatbot.


## Usage 

- Choose a category number to explore questions related to that category.
- Enter a query to get a predicted answer based on the trained model.
- Respond 'yes' or 'no' based on the usefulness of the provided answer.
- Enter 'top5' to get the top 5 most relevant questions for your query.
- Follow on-screen instructions for further interactions.



