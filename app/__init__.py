from flask import Flask, request, jsonify, render_template
from app.main import process_question  # Import your main processing function

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.form['question']
    answer = process_question(user_question)  # Process the question
    return jsonify({'answer': answer})
