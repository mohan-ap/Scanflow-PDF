from flask import Flask, request, jsonify,session
from flask_swagger_ui import get_swaggerui_blueprint
from flask_swagger import swagger
from pymongo import MongoClient
from flask_cors import CORS
import os
from paddleocr import PaddleOCR
import tempfile
from pdf2image import convert_from_path
import numpy as np 
from dotenv import load_dotenv
import secrets
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
import boto3
from langchain.embeddings import OpenAIEmbeddings
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS


load_dotenv()  # take environment variables from .env

app = Flask(__name__)
cors = CORS(app)
app.secret_key = "123"

os.environ["OPENAI_API_KEY"] =os.getenv("OPEN_API_KEY")

ocr = PaddleOCR(use_angle_cls=True,use_gpu=False, lang='en')

embeddings = OpenAIEmbeddings()

chat_history = []

client = MongoClient(host="localhost",port=27017)
db = client['qa']
files=db['files']
history = db['conversations']
users = db ['users']

s3 = boto3.client('s3',
                    aws_access_key_id="AKIA4SSHERTEX2UZYAES",
                    aws_secret_access_key= "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",
                     )

BUCKET_NAME='pdfocrflask'

@app.route('/', methods=['GET'])
def home():
    return "welcome"

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    user_name = data['username']
    password = data['password']

    if user_name and password:
        existing_user = db.users.find_one({'username': user_name})
        if existing_user:
            return jsonify({'message': 'Username already exists'}), 400

        users.insert_one({'username': user_name, 'password': password})

        return jsonify({'message': 'User registered successfully', 'username': user_name}), 200
    else:
        return jsonify({'message': 'Missing username or password'}), 400
    
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user_name = data['username']
    password = data['password']

    if user_name and password:
        user = users.find_one({'username': user_name, 'password': password})
        if user:
            session['user_id'] = user['username']
            print(session['user_id'])
            return jsonify({'message': 'Login successful', 'username': str(user['username'])}), 200
        else:
            return jsonify({'message': 'Invalid username or password'}), 401
    else:
        return jsonify({'message': 'Missing username or password'}), 400
    
@app.route("/upload_file", methods=["POST"])
def process_request():
    global docsearch, uploaded_pdf_data
    try:
        pdf_file = request.files["pdf"]
        uploaded_file_name = pdf_file.filename
        uploaded_pdf_data = pdf_file.read()
        if not uploaded_pdf_data:
            return jsonify({'error': 'Empty file'}),404

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(uploaded_pdf_data)
            pdf_path = temp_pdf.name

        session['file_id'] = secrets.token_hex(16)
        file_id = session['file_id']
        db.files.insert_one({'file_id': file_id, 'file_name': uploaded_file_name})
        pdf_reader = PdfReader(BytesIO(uploaded_pdf_data))
        raw_text = ""
        extracted_text = []
        for page_num in range(len(pdf_reader.pages)):
            with tempfile.TemporaryDirectory() as temp_dir:
                images = convert_from_path(pdf_path, output_folder=temp_dir, first_page=page_num+1, last_page=page_num+1)
                image = np.array(images[0])
                result = ocr.ocr(image, cls=True)
                for idx in range(len(result)):
                    res = result[idx]
                    for line in res:
                        content = line[1][0]
                        content = content.replace("'", "")
                        extracted_text.append(content)
        raw_text = ' '.join(extracted_text)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(raw_text.encode())
            pdf_path = temp_file.name
        uploaded_file = uploaded_file_name.replace('.pdf', '') + '.txt'
        s3.upload_file(
            Bucket=BUCKET_NAME,
            Filename=temp_file.name,
            Key=uploaded_file
        )
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        texts = splitter.create_documents([raw_text])
        docsearch = FAISS.from_documents(texts, embeddings)
        docsearch.save_local(f"{file_id}")

        return jsonify({'message': 'File uploaded successfully',
                        'file_name': '{}'.format(uploaded_file_name), 'file_id': '{}'.format(file_id)}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route("/question/<string:file_id>", methods=["POST"])
def question_request(file_id):
    global chat_history
    try:
        result = db.files.find_one({'file_id': file_id})
        file_name = result['file_name']
        new_db = FAISS.load_local(f"{file_id}", embeddings=embeddings)
        if not new_db:
            return jsonify({'error': 'File not found'}),404

        if 'question' in request.form:
            question = request.form["question"]
            docs = new_db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
            qn_chain = ConversationalRetrievalChain.from_llm(OpenAI(), docs)
            result = qn_chain({"question": question, "chat_history": chat_history})
            answer = result["answer"]
            chat_history.append((question, answer))

            user_id = session.get('user_id')
            if user_id:
                store_messages(user_id, [{'file_name':file_name, 'question':question, 'answer':answer}])

            return jsonify({
                'question': question,
                'answer': answer,
                'chat_history': chat_history
            }),200

    except Exception as e:
        return jsonify({'error': str(e)}),400
    
@app.route("/history", methods=["GET"])
def history():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'User ID not found'})

        conversations = retrieve_conversation(user_id)
        return jsonify(conversations),200

    except Exception as e:
        return jsonify({'error': str(e)}),400
    

def store_messages(user_id, messages):
    conversation = db.history.find_one({'user_id': user_id})

    if conversation:
        conversation['messages'].extend(messages)
        db.history.update_one({'user_id': user_id}, {'$set': {'messages': conversation['messages']}})
    else:
        conversation = {'user_id': user_id, 'messages': messages}
        print(conversation)
        db.history.insert_one(conversation)

def retrieve_conversation(user_id):
    print("before")
    conversation = db.history.find_one({'user_id': user_id})
    print(conversation)
    if conversation:
        return conversation['messages']
    else:
      return jsonify({"message":"no history found"})
      

SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "Scanflow"}
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

