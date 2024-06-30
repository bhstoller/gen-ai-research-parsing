from flask import Flask, request, jsonify, render_template
import logging
from app.main import process_question  # Import your main processing function

app = Flask(__name__)

# Set up basic logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def home():
    app.logger.debug("Home page accessed")
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.form['question']
    app.logger.debug(f"Received question: {user_question}")
    answer = process_question(user_question)  # Process the question
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.logger.debug("Starting Flask app...")
    app.run(debug=True)
