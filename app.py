from flask import Flask, render_template, request, jsonify
import importlib.util
import numpy as np
import sys
import os

app = Flask(__name__)

# This safely loads your file named literally ".py" without changing your code
spec = importlib.util.spec_from_file_location("my_model", os.path.abspath(".py"))
my_model = importlib.util.module_from_spec(spec)
sys.modules["my_model"] = my_model
spec.loader.exec_module(my_model)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message", "").strip().lower()
    
    if not user_message:
        return jsonify({"reply": "Please type something!"})

    # Uses your exact 'vec' layer from your script
    test = my_model.vec(np.array([user_message]))
    
    # Uses your exact trained 'model' from your script
    scores = my_model.model.predict(test, verbose=0)[0]
    best = np.argmax(scores)
    
    # Grabs the answer from your exact 'responses' list
    reply = my_model.responses[best]
    confidence = scores[best]

    return jsonify({
        "reply": f"{reply} (Neural Confidence: {confidence*100:.1f}%)"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)