from flask import Flask, render_template, request, session, flash, redirect, jsonify
import pymysql
import nltk
import random
import os
import pickle
import warnings
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from werkzeug.security import generate_password_hash, check_password_hash
from pymysql.cursors import DictCursor

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure NLTK resources
warnings.filterwarnings('ignore')
nltk.download(['punkt', 'wordnet', 'stopwords', 'punkt_tab', 'averaged_perceptron_tagger', 'omw-1.4'])

# Database configuration
def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='6897',
        database='register',
        cursorclass=DictCursor
    )

# Chatbot intents and AI setup
intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "Good day", "How are you?"],
            "responses": ["Hello! Welcome to CareerGuide Pro!", "Good to see you!", "Hi there, how can I help?"],
            "context_set": ""
        },
        {
            "tag": "farewell",
            "patterns": ["Goodbye", "Bye", "See you later", "Talk to you later"],
            "responses": ["Goodbye! Come back anytime!", "See you soon!", "Have a great day!"],
            "context_set": ""
        },
        # Add all your other intents here
    ]
}

# Load or train chatbot model
try:
    vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))
    model = pickle.load(open('model/chatbot_model.pkl', 'rb'))
except:
    # Training pipeline
    text_data = []
    labels = []
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            tokens = nltk.word_tokenize(pattern.lower())
            filtered_tokens = [lemmatizer.lemmatize(token) 
                             for token in tokens if token.isalpha() and token not in stop_words]
            text_data.append(' '.join(filtered_tokens))
            labels.append(intent['tag'])

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_data)
    y = labels

    model = LogisticRegression(C=1, max_iter=1000, penalty='l2', solver='liblinear')
    model.fit(X, y)

    # Save model
    if not os.path.exists('model'):
        os.makedirs('model')
    
    with open('model/chatbot_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('model/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

# Application routes
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/forgot')
def forgot():
    return render_template('forgot.html')

@app.route('/add_user', methods=['POST'])
def add_user():
    name = request.form.get('name')
    email = request.form.get('uemail')
    password = request.form.get('upassword')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if not all([name, email, password]):
            flash('All fields are required!', 'danger')
            return redirect('/register')

        try:
            hashed_pw = generate_password_hash(password)
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
                    (name, email, hashed_pw)
                )
                conn.commit()
                flash('Registration successful! Please login', 'success')
                return redirect('/')
        except pymysql.IntegrityError:
            flash('Email already registered!', 'danger')
        except Exception as e:
            flash('Registration failed!', 'danger')
            print(f"Database error: {str(e)}")
        finally:
            if 'conn' in locals():
                conn.close()

    return render_template('register.html')

@app.route('/login_validation', methods=['POST'])
def login_validation():
    email = request.form.get('email', '').strip()
    password = request.form.get('password', '').strip()

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            return redirect('/index')
        else:
            flash('Invalid email or password', 'danger')
    except Exception as e:
        flash('Login error occurred', 'danger')
        print(f"Error: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

    return redirect('/')

@app.route('/index')
def index():
    if 'user_id' not in session:
        flash('Please login first', 'danger')
        return redirect('/')
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        data = request.get_json()
        user_input = data.get('message', '')
        
        # Process input
        input_text = vectorizer.transform([user_input])
        prediction = model.predict(input_text)[0]
        
        # Find matching intent
        for intent in intents['intents']:
            if intent['tag'] == prediction:
                response = random.choice(intent['responses'])
                return jsonify({'response': response})
        
        return jsonify({'response': "I'm still learning. Can you rephrase that?"})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/logout')
def logout():
    session.pop('id', None)
    flash("✅ You have been logged out.")
    return redirect('/')



@app.route('/suggestion', methods=['POST'])
def suggestion():
    email = request.form.get('uemail')
    suggesMess = request.form.get('message')

    conn = get_db_connection()
    if conn:
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO suggestion (email, message) VALUES (%s, %s)", (email, suggesMess))
            conn.commit()
            cur.close()
            conn.close()
            flash('✅ Your suggestion has been sent!')
            return redirect('/index')
        except mysql.connector.Error as err:
            print(f"❌ Database error: {err}")
            flash("❌ Failed to save suggestion!")
            return redirect('/')
    else:
        flash("❌ Database connection failed!")
        return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)