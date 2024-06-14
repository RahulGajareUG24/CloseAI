from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from flask import session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

from pathlib import Path as p
from os import environ
from milvus import default_server
from pymilvus import connections
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Milvus
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI

db = SQLAlchemy()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sqrqzq'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:Rahul%406967g@localhost/userdatabase'
db.init_app(app)

# Milvus and Chatbot Initialization
default_server.start()
MILVUS_HOST = '127.0.0.1'
MILVUS_PORT = default_server.listen_port
default_server.set_base_dir('milvus_data')
connections.connect(host='127.0.0.1', port=default_server.listen_port)

OPENAI_API_KEY = "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"
environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

question_prompt_template = "Please provide a summary of the following text. TEXT: {text} SUMMARY:"
question_prompt = PromptTemplate(template=question_prompt_template, input_variables=["text"])

refine_prompt_template = "Write a concise summary of the following text delimited by triple backquotes. Return your response in bullet points which covers the key points of the text. ```{text}```"
refine_prompt = PromptTemplate(template=refine_prompt_template, input_variables=["text"])

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        school = request.form['school']
        password = generate_password_hash(request.form['password'], method='sha256')
        email = request.form['email']

        sql = text("SELECT * FROM users WHERE username = :username")
        user = db.session.execute(sql, {'username': username}).fetchone()
        
        if user:
            flash('User name already exists!')
            return redirect(url_for('register'))

        sql = text("SELECT MAX(userid) AS max_id FROM users")
        last_user = db.session.execute(sql).fetchone()
        
        if last_user.max_id:
            new_userid = last_user.max_id + 1
        else:
            new_userid = 1

        sql = text("""
            INSERT INTO users (userid, username, school, password, email) 
            VALUES (:userid, :username, :school, :password, :email)
        """)
        db.session.execute(sql, {'userid': new_userid, 'username': username, 'school': school, 'password': password, 'email': email})
        db.session.commit()

        flash('Registration successful!')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        sql = text("SELECT * FROM users WHERE email = :email")
        user = db.session.execute(sql, {'email': email}).fetchone()
        
        if not user or not check_password_hash(user.password, password):
            flash('Please check your login details and try again.')
            return redirect(url_for('login'))

        session['user_id'] = user.userid
        return redirect(url_for('mainpage'))
    
    return render_template('login.html')

@app.route('/universities')
def universities():
    return render_template('universities.html')

@app.route('/mainpage')
def mainpage():
    if 'user_id' not in session:
        flash('Please login first.')
        return redirect(url_for('login'))

    return render_template('mainpage.html')

@app.route('/index')
def index():
    university_name = request.args.get('university_name')
    if not university_name:
        flash('Please select a university first.')
        return redirect(url_for('universities'))
    return render_template('index.html', university_name=university_name)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data['query']
    university_name = data.get('university_name')

    if not university_name:
        return jsonify({'error': 'University name is required'}), 400

    MILVUS_COLLECTION_NAME = university_name
    vector_db = Milvus(
        embedding_function=embeddings,
        collection_name=MILVUS_COLLECTION_NAME,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT}
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
    )
    response = qa_chain.run(query)
    
    return jsonify({'answer': response})

@app.route('/summarise', methods=['POST'])
def summarise():
    data = request.json
    university_name = data.get('university_name')

    if not university_name:
        return jsonify({'error': 'University name is required'}), 400

    pdf_file = f'app/content/{university_name}.pdf'
    
    pdf_loader = PyPDFLoader(pdf_file)
    pages = pdf_loader.load_and_split()
    
    llm = OpenAI(model_name="gpt-3.5-turbo")
    refine_chain = load_summarize_chain(
        llm,
        chain_type="refine",
        question_prompt=question_prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
    )

    refine_outputs = refine_chain({"input_documents": pages})
    response = []
    for doc, out in zip(refine_outputs["input_documents"], refine_outputs["intermediate_steps"]):
        summary_info = {
            "summary": out,
            "page_number": doc.metadata["page"]
        }
        response.append(summary_info)

    return jsonify({'summary': response})

@app.route('/run-indexer', methods=['GET'])
def run_indexer():
    directory = 'app/content/'
    loader = DirectoryLoader(directory)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    docs = text_splitter.split_documents(docs)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    MILVUS_COLLECTION_NAME = "Ashoka_university"
    vector_db: Milvus = Milvus.from_documents(
        docs,
        embedding=embeddings,
        collection_name=MILVUS_COLLECTION_NAME,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT}
    )
    return jsonify({"status": "Indexing complete!"})

if __name__ == '__main__':
    app.run(debug=True)
    default_server.stop()
