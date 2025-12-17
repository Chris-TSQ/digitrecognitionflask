from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64
import sqlite3
from datetime import datetime
import os

# Initialize Flask application
app = Flask(__name__)

# Enable CORS for local development
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Database configuration
DATABASE = 'predictions.db'

# Global variable to store trained model
model = None

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            predicted_digit INTEGER NOT NULL,
            confidence REAL NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            image_data TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✓ Database initialized successfully")

def create_model():
    print("Creating and training model on MNIST dataset...")
    print("This may take 1-2 minutes on first run...")
    
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    
    # Build the neural network
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        
        keras.layers.Dropout(0.2),
        
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model with optimizer and loss function
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    print("Training neural network...")
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate on test set to see how well it generalizes
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n✓ Model trained successfully!")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    
    model.save('digit_recognition_model.h5')
    print("✓ Model saved to digit_recognition_model.h5")
    
    return model

def load_or_create_model():
    model_path = 'digit_recognition_model.h5'
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        try:
            model = keras.models.load_model(model_path)
            print("✓ Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new model instead...")
            return create_model()
    else:
        print("No existing model found. Creating new model...")
        return create_model()

def preprocess_image(image_data):
    try:
        # Remove data URL prefix if present
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        image_bytes = base64.b64decode(image_data)
        
        image = Image.open(io.BytesIO(image_bytes))
        
        image = image.convert('L')
        
        image = image.resize((28, 28), Image.LANCZOS)
        
        img_array = np.array(image)
        
        img_array = 255 - img_array
        
        img_array = img_array.astype('float32') / 255.0
        
        img_array = img_array.reshape(1, 784)
        
        return img_array
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Digit Recognition API is running',
        'endpoints': {
            'predict': '/api/predict [POST]',
            'history': '/api/history [GET]',
            'stats': '/api/stats [GET]'
        }
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict_digit():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'error': 'No image data provided'
            }), 400
        
        # Preprocess the image
        processed_image = preprocess_image(data['image'])        
        predictions = model.predict(processed_image, verbose=0)        
        predicted_digit = int(np.argmax(predictions[0]))        
        confidence = float(predictions[0][predicted_digit])        
        all_probabilities = predictions[0].tolist()
        
        try:
            conn = get_db_connection()
            conn.execute(
                'INSERT INTO predictions (predicted_digit, confidence, image_data) VALUES (?, ?, ?)',
                (predicted_digit, confidence, data['image'][:100])  # Store first 100 chars of image
            )
            conn.commit()
            conn.close()
        except Exception as db_error:
            print(f"Database error (non-critical): {db_error}")
        
        # Return prediction result
        return jsonify({
            'digit': predicted_digit,
            'confidence': confidence,
            'probabilities': all_probabilities,
            'message': 'Prediction successful'
        }), 200
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'error': 'Error processing image',
            'details': str(e)
        }), 500

@app.route('/api/history', methods=['GET'])
def get_history():   
    try:
        conn = get_db_connection()
        
        # Get last 50 predictions
        predictions = conn.execute(
            'SELECT id, predicted_digit, confidence, timestamp FROM predictions ORDER BY timestamp DESC LIMIT 50'
        ).fetchall()
        
        conn.close()
        
        # Convert to list of dictionaries
        history = []
        for pred in predictions:
            history.append({
                'id': pred['id'],
                'digit': pred['predicted_digit'],
                'confidence': pred['confidence'],
                'timestamp': pred['timestamp']
            })
        
        return jsonify({
            'history': history,
            'count': len(history)
        }), 200
        
    except Exception as e:
        print(f"History error: {e}")
        return jsonify({
            'error': 'Error retrieving history'
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        conn = get_db_connection()
        
        # Count predictions for each digit
        stats = conn.execute('''
            SELECT predicted_digit, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM predictions
            GROUP BY predicted_digit
            ORDER BY predicted_digit
        ''').fetchall()
        
        # Get total predictions
        total = conn.execute('SELECT COUNT(*) as total FROM predictions').fetchone()
        
        conn.close()
        
        # Format statistics
        digit_stats = []
        for stat in stats:
            digit_stats.append({
                'digit': stat['predicted_digit'],
                'count': stat['count'],
                'avg_confidence': round(stat['avg_confidence'], 3)
            })
        
        return jsonify({
            'total_predictions': total['total'],
            'digit_distribution': digit_stats
        }), 200
        
    except Exception as e:
        print(f"Stats error: {e}")
        return jsonify({
            'error': 'Error retrieving statistics'
        }), 500

# Initialize database and load/create model when app starts
print("\n" + "="*50)
print("Initializing Digit Recognition Backend")
print("="*50)

init_db()
model = load_or_create_model()

print("\n" + "="*50)
print("✓ Backend ready!")
print("API running on http://localhost:5000")
print("="*50 + "\n")

if __name__ == '__main__':
    # Run the Flask development server
    app.run(host='0.0.0.0', port=5000, debug=True)