import os
import io
from PIL import Image
import numpy as np
from flask import Flask, request, render_template, send_from_directory, jsonify
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Contrôler manuellement la mise à l'échelle de l'interface utilisateur
os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"

current_dir = os.getcwd()

app = Flask(__name__)
Folder = os.path.join(current_dir,"Images")
app.config['Folder'] = Folder
DATA = os.path.join(current_dir,"Data_Train")
CLASSES = ['CI', 'CII', 'CII_V', 'CIII', 'CIII_V', 'CIV', 'CIV_V', 'CV']
num_classes = len(CLASSES)
epochs = 20
learning_rate = 0.001

# Parcourir le répertoire des données et charger les images et leurs étiquettes correspondantes
images = []
labels = []

for parent_folder, subfolders, files in os.walk(DATA):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(parent_folder, file)
            try:
                img = Image.open(image_path).convert('RGB')
                img = img.resize((224, 224))
                image_array = np.array(img)
                if image_array.shape == (224, 224, 3):
                    class_name = os.path.basename(parent_folder)
                    images.append(image_array)
                    labels.append(CLASSES.index(class_name))
                else:
                    print(f"Ignored incorrectly sized image - {image_path}")
            except Exception as e:
                print(f"Error loading image - {image_path}: {e}")

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int64)
y_test = np.array(y_test, dtype=np.int64)

# Charger le modèle pré-entrainé
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Sequential()
model.add(base_model)

# Ajouter des couches de convolution personnalisées (commentées)
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# Effectuer la classification
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Fine tuning (geler les couches du modèle pré-entrainé)
for layer in base_model.layers:
    layer.trainable = False

# Compiler et entraîner le modèle
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=epochs, batch_size=32)

loss = history.history['loss']
accuracy = history.history['accuracy']
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {test_loss}, Accuracy: {test_accuracy}')


# Fonction de prédiction
def predict_class(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = img_to_array(img)
    img = img[np.newaxis] / 255.0
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    predicted_class = CLASSES[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    return predicted_class, confidence


# Route pour télécharger une image pour la classification
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config['Folder'], filename)
            uploaded_file.save(file_path)
            predicted_class, confidence = predict_class(file_path)
            return render_template('result.html', predicted_class=predicted_class, confidence=confidence,
                                   image_path=filename)
    return render_template('index.html')


# Route pour afficher les images téléchargées
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['Folder'], filename)


# API de classification
@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'})
    if uploaded_file:
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(app.config['Folder'], filename)
        uploaded_file.save(file_path)
        predicted_class, confidence = predict_class(file_path)
        return jsonify({'class': predicted_class, 'confidence': confidence})


# Affiche des métriques de modèle
@app.route('/metrics', methods=['GET'])
def metrics():
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    confusion = confusion_matrix(y_test, y_pred_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('static/confusion_matrix.png')

    accuracy = accuracy_score(y_test, y_pred_classes)
    classification_report_str = classification_report(y_test, y_pred_classes, target_names=CLASSES, output_dict=False)

    with open('static/classification_report.txt', 'w') as report_file:
        report_file.write(classification_report_str)

    return render_template('metrics.html', accuracy=accuracy, confusion_matrix='confusion_matrix.png',
                           classification_report='classification_report.txt')


if __name__ == '__main__':
    app.run(debug=True)
