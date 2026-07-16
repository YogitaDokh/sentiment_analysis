import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the vectorizer and model safely
try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model (1).pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model and Vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    vectorizer = None
    model = None

# Attractive, UI/UX rich Dashboard Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Text Sentiment Analyzer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 15px rgba(99, 102, 241, 0.2); }
            50% { box-shadow: 0 0 25px rgba(99, 102, 241, 0.6); }
        }
        .glow-btn { animation: pulse-glow 3s infinite; }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col justify-between font-sans selection:bg-indigo-500 selection:text-white">

    <header class="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
        <div class="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="bg-indigo-600 p-2 rounded-xl text-white shadow-lg shadow-indigo-600/30">
                    <i class="fa-solid fa-brain text-xl animate-pulse"></i>
                </div>
                <div>
                    <h1 class="text-lg font-bold tracking-tight bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">SentixAI</h1>
                    <p class="text-xs text-slate-400">NLP Classification Engine</p>
                </div>
            </div>
            <span class="flex items-center bg-emerald-500/10 text-emerald-400 text-xs px-2.5 py-1 rounded-full border border-emerald-500/20 font-medium">
                <span class="w-2 h-2 bg-emerald-400 rounded-full mr-1.5 animate-ping"></span> Live & Active
            </span>
        </div>
    </header>

    <main class="max-w-3xl mx-auto px-4 py-12 flex-grow w-full">
        <div class="text-center mb-10">
            <h2 class="text-3xl md:text-4xl font-extrabold tracking-tight text-white mb-3">
                Analyze Text Sentiment Instantly
            </h2>
            <p class="text-slate-400 max-w-xl mx-auto text-sm md:text-base">
                Type or paste your text below. Our machine learning model will evaluate whether the underlying intent is positive or negative.
            </p>
        </div>

        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 md:p-8 shadow-2xl relative overflow-hidden">
            <div class="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500"></div>

            <form id="analysisForm" class="space-y-6">
                <div>
                    <label for="text" class="block text-sm font-semibold text-slate-300 mb-2 flex justify-between">
                        <span>Input Text</span>
                        <span id="charCount" class="text-xs text-slate-500 font-normal">0 characters</span>
                    </label>
                    <textarea 
                        id="text" 
                        name="text" 
                        rows="4" 
                        required
                        placeholder="Type something honest, happy, or critical..." 
                        class="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-200 placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition duration-200 resize-none"
                    ></textarea>
                </div>

                <button 
                    type="submit" 
                    id="submitBtn"
                    class="glow-btn w-full bg-indigo-600 hover:bg-indigo-500 text-white font-semibold py-3.5 px-6 rounded-xl shadow-lg transition duration-200 flex items-center justify-center space-x-2 group active:scale-[0.99]"
                >
                    <span>Analyze Intent</span>
                    <i class="fa-solid fa-arrow-right text-xs transition-transform group-hover:translate-x-1" id="btnIcon"></i>
                </button>
            </form>

            <div id="resultContainer" class="mt-8 pt-8 border-t border-slate-800/60 hidden transition-all duration-300">
                <div id="resultCard" class="rounded-xl p-5 border flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                    <div class="flex items-center space-x-4">
                        <div id="statusIconContainer" class="p-3 rounded-xl text-xl">
                            <i id="statusIcon" class="fa-solid"></i>
                        </div>
                        <div>
                            <p class="text-xs tracking-wider uppercase font-semibold text-slate-400">Predicted Emotion</p>
                            <h3 id="predictionText" class="text-2xl font-bold uppercase tracking-wide"></h3>
                        </div>
                    </div>
                    
                    <div class="w-full md:w-48 bg-slate-950/80 p-3 rounded-lg border border-slate-800">
                        <div class="flex justify-between text-xs mb-1 font-medium">
                            <span class="text-slate-400">Confidence Score</span>
                            <span id="confidenceValue" class="text-indigo-400 font-bold">0%</span>
                        </div>
                        <div class="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                            <div id="confidenceBar" class="h-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all duration-500" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <footer class="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-600">
        <p>&copy; 2026 SentixAI. Powered by Scikit-Learn Engine.</p>
    </footer>

    <script>
        const textInput = document.getElementById('text');
        const charCount = document.getElementById('charCount');
        const form = document.getElementById('analysisForm');
        const submitBtn = document.getElementById('submitBtn');
        const btnIcon = document.getElementById('btnIcon');
        const resultContainer = document.getElementById('resultContainer');
        const resultCard = document.getElementById('resultCard');
        const statusIconContainer = document.getElementById('statusIconContainer');
        const statusIcon = document.getElementById('statusIcon');
        const predictionText = document.getElementById('predictionText');
        const confidenceValue = document.getElementById('confidenceValue');
        const confidenceBar = document.getElementById('confidenceBar');

        // Dynamic Character Counter
        textInput.addEventListener('input', (e) => {
            charCount.textContent = `${e.target.value.length} characters`;
        });

        // Form Submit Handler
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // Set loading state
            submitBtn.disabled = true;
            submitBtn.classList.add('opacity-75', 'cursor-not-allowed');
            submitBtn.querySelector('span').textContent = 'Processing Matrix...';
            btnIcon.className = 'fa-solid fa-circle-notch animate-spin text-xs';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: textInput.value })
                });
                
                const data = await response.json();
                
                if(data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                // Render Response Beautifully
                resultContainer.classList.remove('hidden');
                predictionText.textContent = data.prediction;
                
                const percentage = (data.confidence * 100).toFixed(1);
                confidenceValue.textContent = `${percentage}%`;
                confidenceBar.style.width = `${percentage}%`;

                if(data.prediction.toLowerCase() === 'positive') {
                    // Positive Themes (Emerald / Green)
                    resultCard.className = 'rounded-xl p-5 border border-emerald-500/30 bg-emerald-500/5 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 transition-all duration-300';
                    statusIconContainer.className = 'p-3 rounded-xl text-xl bg-emerald-500/20 text-emerald-400';
                    statusIcon.className = 'fa-solid fa-face-smile';
                    predictionText.className = 'text-2xl font-bold uppercase tracking-wide text-emerald-400';
                } else {
                    // Negative Themes (Rose / Red)
                    resultCard.className = 'rounded-xl p-5 border border-rose-500/30 bg-rose-500/5 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 transition-all duration-300';
                    statusIconContainer.className = 'p-3 rounded-xl text-xl bg-rose-500/20 text-rose-400';
                    statusIcon.className = 'fa-solid fa-face-frown';
                    predictionText.className = 'text-2xl font-bold uppercase tracking-wide text-rose-400';
                }
                
                // Smooth scroll down to view result
                resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

            } catch (error) {
                console.error('Error fetching prediction:', error);
                alert('Connection error. Is the server running?');
            } finally {
                // Reset loading state
                submitBtn.disabled = false;
                submitBtn.classList.remove('opacity-75', 'cursor-not-allowed');
                submitBtn.querySelector('span').textContent = 'Analyze Intent';
                btnIcon.className = 'fa-solid fa-arrow-right text-xs transition-transform group-hover:translate-x-1';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

# API Endpoint (Handles asynchronous Javascript requests seamlessly)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field in request body'}), 400
    
    if not model or not vectorizer:
        return jsonify({'error': 'Model or Vectorizer components missing on backend'}), 500

    text = data['text']
    
    # ML Pipeline Execution
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)[0]
    probabilities = model.predict_proba(transformed_text)[0]
    
    # Calculate exact confidence based on target class mapping
    class_idx = 1 if prediction == 'positive' else 0
    confidence = float(probabilities[class_idx])

    return jsonify({
        'text': text,
        'prediction': str(prediction),
        'confidence': confidence
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
