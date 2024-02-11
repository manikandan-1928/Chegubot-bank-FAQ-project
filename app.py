from flask import Flask, render_template, request
from src.loader import load_all_data
from src.utils import *
from src.preprocess import TextPreprocessor
from src.answer_predict import Chatbot
import datetime

app = Flask(__name__)
app.static_folder = 'static'

# Load necessary data and models
data, tokenizer, bert_model = load_all_data('tokenizer', 'model', 'data/embedded_data.csv')

# Define the category list and calculate class indices
category = '''1 : security
             2 : loans
             3 : accounts
             4 : insurance
             5 : investments
             6 : fundstransfer
             7 : cards'''

class_indices = calculate_class_indices(data)

# Initialize TextPreprocessor and Chatbot instances
qn_cleaning = TextPreprocessor()
chatbot = Chatbot(data, qn_cleaning, tokenizer, bert_model)

# Initialize global variables to store context for user session
selected_class = None
top_5_qns = None
top_5_ans = None
filtered_data = None

@app.route('/')
def home():
    # Render the home page with category options
    current_time = datetime.datetime.now().strftime("%H:%M")
    return render_template('index.html', messages=category, current_time=current_time)

@app.route('/get', methods=['GET'])
def get_bot_response():
    global selected_class, top_5_qns, top_5_ans, filtered_data
    user_query = request.args.get('msg', '')
    messages = []

    if user_query.isdigit() and 1 <= int(user_query) <= 7:
        # User selected a category number
        top_5 = False
        selected_class_idx = int(user_query)
        filtered_data, selected_class = filter_data_by_class(selected_class_idx, data)

        if filtered_data is not None:
            print(filtered_data.head(1))
            print(selected_class)
            messages.append('Please ask your query.')
        else:
            messages.append('Invalid category number')

    elif isinstance(user_query, str) and len(user_query) > 5:
        # User entered a query
        print(selected_class)
        print(filtered_data.head(1))

        predicted_question, predicted_answer, top_5_qns, top_5_ans = chatbot.predict_answer(user_query, selected_class, filtered_data, class_indices)
        messages.append(predicted_answer)
        messages.append('Was this answer useful? (yes/no)')

    elif user_query == 'yes':
        # User found the answer useful
        message = 'Good to hear. If you have any other query, please choose the category number.'
        messages.append(message)
        messages.append(category)

    elif user_query == 'no':
        # User didn't find the answer useful
        message = 'To provide 5 most relevant questions, please enter - top5'
        messages.append(message)

    elif user_query.lower() == 'top5':
        # User requested the top 5 questions
        messages.append('Please find the most relevant 5 questions based on your query')
        print(top_5_qns)

        # Display top 5 questions with alphabets
        for idx, qn in enumerate(top_5_qns):
            alphabet = chr(ord('a') + idx)
            messages.append(f'{alphabet} - {qn}')

        messages.append('Please provide the respective alphabet to get the answer.')

    elif isinstance(user_query, str) and user_query.lower() in ['a', 'b', 'c', 'd', 'e'] and len(user_query) == 1:
        # User selected an answer from the top 5
        idx = ord(user_query.lower()) - ord('a')
        selected_answer = top_5_ans[idx]
        messages.append(selected_answer)
        messages.append('Hope it clears your doubt. If you have any other query, please choose the category number.')
        messages.append(category)

    elif user_query.lower() == 'q':
        # User decided to quit
        messages.append('Bye buddy! Take care and have a nice day.')

    else:
        # User entered an invalid input
        messages.append('Invalid input. Please enter a valid query or category number.')

    # Convert the list to a multi-line string with double line breaks
    messages_str = '\n\n'.join(messages)
    print(messages_str)
    return render_template('index.html', messages=messages_str)

if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True)
