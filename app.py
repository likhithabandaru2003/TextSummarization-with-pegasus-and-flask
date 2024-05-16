from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import os
from io import BytesIO
from docx import Document
from pdfminer import high_level
from bs4 import BeautifulSoup
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Api(app)

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine('postgresql://postgres:postgres@localhost/textsummarization')
Session = scoped_session(sessionmaker(bind=engine))

class DocumentSummary(Base):
    __tablename__ = 'document_summaries'
    id = Column(Integer, primary_key=True)
    original_filename = Column(String)
    input_text = Column(Text)
    summary_text = Column(Text)

Base.metadata.create_all(engine)

class TextSummarizer(Resource):
    def __init__(self):
        super().__init__()
        # Initialize Pegasus model and tokenizer
        self.model_name = 'google/pegasus-large'
        self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name)

    def split_and_summarize_chunks(self, input_text):
        # Split the input text into smaller chunks
        chunk_size = 800  # Adjust the chunk size as needed
        chunks = [input_text[i:i + chunk_size] for i in range(0, len(input_text), chunk_size)]

        # Summarize each chunk individually
        summaries = []
        for chunk in chunks:
            inputs = self.tokenizer(chunk, return_tensors='pt', truncation=True, max_length=chunk_size, padding='max_length')
            summary_ids = self.model.generate(inputs['input_ids'], num_beams=5, max_length=150, early_stopping=True)
            summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary_text)

        # Combine the chunk summaries into a cohesive representation of the entire document
        combined_summary = '\n'.join(summaries)
        return combined_summary

    def post(self):
        try:
            # Receive input file and format (DOCX, PDF, TXT, HTML)
            file = request.files['file']
            filename = file.filename
            file_format = filename.rsplit('.', 1)[-1].lower()  # Extract file extension and convert to lowercase

            # Extract text content from different file formats
            if file_format == 'docx':
                doc = Document(BytesIO(file.read()))
                input_text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            elif file_format == 'pdf':
                input_text = high_level.extract_text(BytesIO(file.read()))
            elif file_format == 'txt':
                input_text = file.read().decode('utf-8')
            elif file_format == 'html':
                soup = BeautifulSoup(file.read(), 'html.parser')
                input_text = soup.get_text()

            # Split and summarize chunks of the input text
            combined_summary = self.split_and_summarize_chunks(input_text)

            # Save input document and summary in the database
            session = Session()
            document_summary = DocumentSummary(original_filename=file.filename, input_text=input_text, summary_text=combined_summary)
            session.add(document_summary)
            session.commit()
            
            # Get the ID of the saved object
            saved_id = document_summary.id
            
            session.close()

            # Return the ID along with input text and summary in the response
            return {'id': saved_id, 'input_text': input_text, 'summary_text': combined_summary}

        except Exception as e:
            # Log the exception for debugging
            print(f"Exception occurred: {e}")
            # Return an error response
            return {'error': 'An error occurred during summarization process.'}, 500

api.add_resource(TextSummarizer, '/summarize')

# Add a new route to fetch data using ID
class SummaryById(Resource):
    def get(self, summary_id):
        session = Session()
        summary = session.query(DocumentSummary).filter_by(id=summary_id).first()
        session.close()
        if summary:
            return {'input_text': summary.input_text, 'summary_text': summary.summary_text}
        else:
            return {'error': 'Summary not found.'}, 404

api.add_resource(SummaryById, '/summary/<int:summary_id>')

if __name__ == '__main__':
    app.run(debug=True)
