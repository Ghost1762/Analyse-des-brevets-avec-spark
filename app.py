import os
import re
import secrets
import torch
import schedule
import time
import threading
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, MarianMTModel, MarianTokenizer
import pandas as pd
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from itertools import chain
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from confluent_kafka import Producer, Consumer, KafkaError
from google.cloud import firestore

firestore_client = firestore.Client(project="lithe-window-413300")


app = Flask(__name__, static_url_path='/static')
app.secret_key = secrets.token_hex(16)  # Generate a random secret key

client = MongoClient('localhost', 27017)
db = client['spark']
db_users = client['spark']['users']
db_consultation = client['spark']['consultation']





# Charger les informations de configuration depuis config.json
with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)

smtp_server = config_data["server"]
smtp_port = config_data["port"]
sender_email = config_data["email"]
sender_password = config_data["pwd"]
file_path = '20230714-production-vegetale-2010-2022.xlsx'
df = pd.read_excel(file_path)

# Renommer les colonnes pour enlever les espaces de début et de fin
df.columns = df.columns.str.strip()

@app.route('/query', methods=['POST'])
def query():
    query_type = request.json.get('query_type')
    response = {}

    try:
        if query_type == 'crop_yield':
            result = df[df['Indicateur'] == 'Production-T (Tonnes)'].groupby('Filière')['Valeur'].mean().reset_index()
            print(result)  # Impression pour débogage
            response = {'message': 'Le rendement moyen des cultures par filière est : ' + ', '.join(
                [f'{row["Filière"]}: {row["Valeur"]:.2f}' for index, row in result.iterrows()])}

        elif query_type == 'crop_area':
            result = df[df['Indicateur'] == 'Superficie (Ha)'].groupby('Filière')['Valeur'].mean().reset_index()
            print(result)  # Impression pour débogage
            response = {'message': 'La superficie moyenne des cultures par filière est : ' + ', '.join(
                [f'{row["Filière"]}: {row["Valeur"]:.2f}' for index, row in result.iterrows()])}

        elif query_type == 'total_production_by_year':
            result = df[df['Indicateur'] == 'Production-T (Tonnes)'].groupby('Occurrence')['Valeur'].sum().reset_index()
            print(result)  # Impression pour débogage
            response = {'message': 'La production totale par année est : ' + ', '.join(
                [f'{row["Occurrence"]}: {row["Valeur"]:.2f} tonnes' for index, row in result.iterrows()])}

        elif query_type == 'total_area_by_product':
            result = df[df['Indicateur'] == 'Superficie (Ha)'].groupby('Produit')['Valeur'].sum().reset_index()
            print(result)  # Impression pour débogage
            response = {'message': 'La superficie totale par produit est : ' + ', '.join(
                [f'{row["Produit"]}: {row["Valeur"]:.2f} hectares' for index, row in result.iterrows()])}

        print("Réponse:", response)  # Impression pour débogage
        return jsonify(response)

    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500

sentiment_model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
sentiment_tokenizer = DistilBertTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = DistilBertForSequenceClassification.from_pretrained(sentiment_model_name, num_labels=2)

translation_models = {
    "fr": ("Helsinki-NLP/opus-mt-fr-en", MarianTokenizer, MarianMTModel),
    "ar": ("Helsinki-NLP/opus-mt-ar-en", MarianTokenizer, MarianMTModel),
    "ma": ("Helsinki-NLP/opus-mt-ar-en", MarianTokenizer, MarianMTModel)
}

translators = {}
for lang, (model_name, tokenizer_class, model_class) in translation_models.items():
    tokenizer = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name)
    translators[lang] = (tokenizer, model)



def translate_text(text, lang):
    tokenizer, model = translators[lang]
    encoded_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**encoded_text)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

def predict_sentiment(texts, lang=None):
    if lang and lang != "en":
        texts = [translate_text(text, lang) for text in texts]
    encodings = sentiment_tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = sentiment_model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=-1).tolist()
        labels = ["negative" if pred == 0 else "positive" for pred in predictions]
        return labels

# Middleware pour gérer les en-têtes CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        comment = data['comment']
        lang = data.get('language', 'en')
        processed_comment = re.sub(r'(.)\1+', r'\1', comment)
        prediction = predict_sentiment([processed_comment], lang=lang)
        sentiment = prediction[0]
        return jsonify({"prediction": sentiment})
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('bigdata.html')

@app.route('/login2', methods=['GET', 'POST'])
def index2():
    return render_template('index.html')

@app.route('/login3', methods=['GET', 'POST'])
def index3():
    return render_template('signup.html')

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['pass']

    # Vérifier les informations d'identification dans la base de données
    user = db_users.find_one({'email': email, 'password': password})
    if user:
        # Connectez-vous avec succès, stockez le nom de l'utilisateur dans la session
        session['username'] = user['fullname']
        return redirect(url_for('recherche'))
    else:
        # Informez l'utilisateur que les informations d'identification sont incorrectes
        return render_template('index.html', error_message='Invalid email or password')


@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        fullname = request.form['fullname']
        dob = request.form['dob']
        phone = request.form['phone']
        email = request.form['email']
        password = request.form['pass']

        data = {
            'fullname': fullname,
            'dob': dob,
            'phone': phone,
            'email': email,
            'password': password
        }

        # Save to MongoDB and get the inserted_id
        inserted_id = db_users.insert_one(data).inserted_id

        # Add the MongoDB ObjectId to the data dictionary
        data['_id'] = str(inserted_id)

        # Save to Firestore
        save_to_firestore('users', email, data)

        return render_template('signup.html')

def save_to_firestore(collection_name, document_id, data):
    # Convert any ObjectId to string in the data
    data = convert_objectid(data)
    doc_ref = firestore_client.collection(collection_name).document(document_id)
    doc_ref.set(data)
    print(f'Document {document_id} in collection {collection_name} saved.')

def convert_objectid(data):
    if isinstance(data, dict):
        return {k: convert_objectid(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_objectid(i) for i in data]
    elif isinstance(data, ObjectId):
        return str(data)
    else:
        return data
@app.route('/recherche', methods=['GET', 'POST'])
def recherche():
    if request.method == 'POST':
        selected_ids = request.form.getlist('selectedIds[]')
        search_term = request.form['search']
        selected_ids_str = request.form['selected_ids_str']

        username = session.get('username', None)

        if selected_ids_str == 'google':
            results, total_results = search_in_google_db(search_term, username)
        elif selected_ids_str == 'upsto':
            results, total_results = search_in_upsto_db(search_term, username)
        elif selected_ids_str == 'epo':
            results, total_results = search_in_epo_db(search_term, username)
        elif selected_ids_str == 'wipo':
            results, total_results = search_in_wipo_db(search_term, username)
        else:
            # If none of the specific databases are selected, search in all databases
            google_results, _ = search_in_google_db(search_term, username)
            epo_results, _ = search_in_epo_db(search_term, username)
            wipo_results, _ = search_in_wipo_db(search_term, username)

            # Aggregate results from all databases
            results = list(chain(google_results, epo_results, wipo_results))
            total_results = len(results)

        return render_template('recherche.html', results=results, total_results=total_results)
    return render_template('recherche.html', results=None, total_results=0)

def search_in_google_db(search_term, username=None):
    results = list(db.data.find({"title": {"$regex": search_term, "$options": "i"}}))
    total_results = len(results)
    if username:
        for result in results:
            consultation = db_consultation.find_one({"username": username, "patent_id": result["ID"]})
            if consultation:
                result["consultation_count"] = consultation.get("consultation_count", 0)
                result["last_consulted"] = consultation.get("last_consulted", None)
            else:
                result["consultation_count"] = 0
                result["last_consulted"] = None
    return results, total_results

def search_in_upsto_db(search_term,username=None):
    results = list(db.patentscope.find({"title": {"$regex": search_term, "$options": "i"}}))
    total_results = len(results)
    if username:
        for result in results:
            consultation = db_consultation.find_one({"username": username, "patent_id": result["ID"]})
            if consultation:
                result["consultation_count"] = consultation.get("consultation_count", 0)
                result["last_consulted"] = consultation.get("last_consulted", None)
            else:
                result["consultation_count"] = 0
                result["last_consulted"] = None
    return results, total_results

def search_in_epo_db(search_term,username=None):
    results = list(db.FPO.find({"title": {"$regex": search_term, "$options": "i"}}))
    total_results = len(results)
    if username:
        for result in results:
            consultation = db_consultation.find_one({"username": username, "patent_id": result["ID"]})
            if consultation:
                result["consultation_count"] = consultation.get("consultation_count", 0)
                result["last_consulted"] = consultation.get("last_consulted", None)
            else:
                result["consultation_count"] = 0
                result["last_consulted"] = None
    return results, total_results

def search_in_wipo_db(search_term,username=None):
    results = list(db.wip.find({"title": {"$regex": search_term, "$options": "i"}}))
    total_results = len(results)
    if username:
        for result in results:
            consultation = db_consultation.find_one({"username": username, "patent_id": result["ID"]})
            if consultation:
                result["consultation_count"] = consultation.get("consultation_count", 0)
                result["last_consulted"] = consultation.get("last_consulted", None)
            else:
                result["consultation_count"] = 0
                result["last_consulted"] = None
    return results, total_results

from flask import request

@app.route('/detail/<patent_id>')
def detail(patent_id):
    collection = request.args.get('collection')  # Récupérer la valeur de la collection de l'URL
    if collection == 'data':
        patent_details = db.data.find_one({"ID": patent_id})
    elif collection == 'FPO':
        patent_details = db.FPO.find_one({"ID": patent_id})
    elif collection == 'wip':
        patent_details = db.wip.find_one({"ID": patent_id})
    else:
        patent_details = db.data.find_one({"ID": patent_id})

    current_time = datetime.now()

    if 'username' in session:
        username = session['username']
        db_consultation = db['consultation']  # Assurez-vous d'avoir une collection de consultation
        db_consultation.update_one(
            {"username": username, "patent_id": patent_id},
            {"$inc": {"consultation_count": 1}, "$set": {"last_consulted": current_time}},
            upsert=True
        )

    return render_template('detail.html', patent_details=patent_details)


@app.route('/historique')
def historique():
    if 'username' in session:
        username = session['username']
        consultations = db_consultation.find({"username": username}, {"patent_id": 1, "last_consulted": 1}).sort("last_consulted", -1)
        consultations_dict = {str(consultation["patent_id"]): consultation["last_consulted"] for consultation in consultations}
        return jsonify(consultations_dict)
    else:
        return jsonify({})

@app.route('/historique/delete/<patent_id>', methods=['DELETE'])
def delete_from_historique(patent_id):
    if 'username' in session:
        username = session['username']
        db_consultation.delete_one({"username": username, "patent_id": patent_id})
        return jsonify({"message": "L'élément a été supprimé avec succès"})
    else:
        return jsonify({"error": "Vous devez être connecté pour effectuer cette action"}), 401

@app.route('/user_info', methods=['GET'])
def user_info():
    if 'username' in session:
        username = session['username']
        user_data = db_users.find_one({'fullname': username})
        if user_data:
            html = f'<div>'
            html += f'<h5>Nom complet:</h5><p class="lead">{user_data["fullname"]}</p>'
            html += f'<h5>Date de naissance:</h5><p class="lead">{user_data["dob"]}</p>'
            html += f'<h5>Téléphone:</h5><p class="lead">{user_data["phone"]}</p>'
            html += f'<h5>Email:</h5><p class="lead">{user_data["email"]}</p>'
            html += '</div>'
            return html
        else:
            return '<div class="container">Informations utilisateur non disponibles</div>'
    else:
        return '<div class="container">Utilisateur non connecté</div>'

KAFKA_TOPIC = 'email-topic'
KAFKA_BROKER = 'localhost:9092'

# Initialisation du producteur Kafka
producer_config = {
    'bootstrap.servers': KAFKA_BROKER
}
producer = Producer(**producer_config)

def schedule_email():
    for user in db_users.find():
        message = {
            "event": "send_email",
            "recipient": user['email'],
            "fullname": user['fullname']
        }
        producer.produce(KAFKA_TOPIC, value=json.dumps(message))
    producer.flush()
    print("Scheduled email event sent to Kafka")

def run_scheduler():
    schedule.every(60).minutes.do(schedule_email)
    while True:
        schedule.run_pending()
        time.sleep(1)

# Lancer le scheduler dans un thread séparé
scheduler_thread = threading.Thread(target=run_scheduler)
scheduler_thread.start()

# Kafka Consumer
consumer_config = {
    'bootstrap.servers': KAFKA_BROKER,
    'group.id': 'flask-email-group',
    'auto.offset.reset': 'earliest'
}
consumer = Consumer(**consumer_config)
consumer.subscribe([KAFKA_TOPIC])

# Fonction pour envoyer des emails
def send_email(recipient, subject, body):
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient
    message['Subject'] = subject

    message.attach(MIMEText(body, 'plain'))

    with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient, message.as_string())

def consume_messages():
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            else:
                print(msg.error())
                break
        message = json.loads(msg.value().decode('utf-8'))
        if message['event'] == 'send_email':
            with app.app_context():
                recipient = message.get('recipient')
                fullname = message.get('fullname')
                if not recipient or not fullname:
                    print("Missing recipient or fullname in message")
                    continue

                random_patent = db.data.aggregate([{'$sample': {'size': 1}}]).next()
                patent_detail_link = f"http://127.0.0.1:5000/detail/{random_patent['ID']}"
                email_content = f"""
                Bonjour {fullname},

                Voici les informations d'un brevet aléatoire :

                Titre : {random_patent['title']}
                ID : {random_patent['ID']}
                {'Inventeurs : ' + ', '.join(random_patent['inventors']) if 'inventors' in random_patent else ''}
                {'Date de publication : ' + random_patent['publication_date'] if 'publication_date' in random_patent else ''}
                {'Pays : ' + random_patent['country'] if 'country' in random_patent else ''}
                {'Assignataires actuels : ' + ', '.join(random_patent['current_assignees']) if 'current_assignees' in random_patent else ''}
                {'Date de priorité : ' + random_patent['priority_date'] if 'priority_date' in random_patent else ''}
                {'Langue : ' + random_patent['other_language'] if 'other_language' in random_patent else ''}

                Vous pouvez consulter ce brevet en détail sur notre site : {patent_detail_link}.

                Cordialement,
                Votre équipe AGROPA
                """
                send_email(recipient, "Nouvelle alerte de brevet", email_content)

# Lancer le consommateur Kafka dans un thread séparé
consumer_thread = threading.Thread(target=consume_messages)
consumer_thread.start()


@app.route('/send_email', methods=['GET'])
def send_email_route():
    if 'username' in session:
        username = session['username']
        user_data = db_users.find_one({'fullname': username})
        if user_data:
            random_patent = db.data.aggregate([{'$sample': {'size': 1}}]).next()
            patent_detail_link = f"http://127.0.0.1:5000/detail/{random_patent['ID']}"
            email_content = f"""
            Bonjour {user_data['fullname']},

            Voici les informations d'un brevet aléatoire :

            Titre : {random_patent['title']}
            ID : {random_patent['ID']}
            {'Inventeurs : ' + ', '.join(random_patent['inventors']) if 'inventors' in random_patent else ''}
            {'Date de publication : ' + random_patent['publication_date'] if 'publication_date' in random_patent else ''}
            {'Pays : ' + random_patent['country'] if 'country' in random_patent else ''}
            {'Assignataires actuels : ' + ', '.join(random_patent['current_assignees']) if 'current_assignees' in random_patent else ''}
            {'Date de priorité : ' + random_patent['priority_date'] if 'priority_date' in random_patent else ''}
            {'Langue : ' + random_patent['other_language'] if 'other_language' in random_patent else ''}

            Vous pouvez consulter ce brevet en détail sur notre site : {patent_detail_link}.

            Cordialement,
            Votre équipe AGROPA
            """

            user_email = user_data['email']
            send_email(user_email, "Nouvelle alerte de brevet", email_content)

            return jsonify({'message': 'L\'e-mail a été envoyé avec succès'})
        else:
            return jsonify({'error': 'Informations utilisateur non disponibles'}), 400
    else:
        return jsonify({'error': 'Utilisateur non connecté'}), 401


def convert_objectid(data):
    if isinstance(data, dict):
        return {k: convert_objectid(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_objectid(i) for i in data]
    elif isinstance(data, ObjectId):
        return str(data)
    else:
        return data




if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
