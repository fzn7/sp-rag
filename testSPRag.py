#!/usr/bin/env python
# coding: utf-8

import os
import requests
from flask import Flask, request, jsonify

from sprag.knowledge_base import KnowledgeBase
from sprag.reranker import NoReranker
from langchain_openai import ChatOpenAI

reranker = NoReranker()

BOT_TOKEN = os.environ["BOT_TOKEN"]


kb_id = "test_kb_id"
kb = KnowledgeBase(kb_id, reranker=reranker, storage_directory="./data_tmp")
from langchain_core.prompts import PromptTemplate

template = """
Ты ассистент. 
Когда пользователь задаст вопрос необходимо дать на него ответ и приложить выдержки из твоей базы знаний. 
Держи ответ в рамках пары предложений:

База знаний: {context}

Отвечай в формате:

Ответ:
 - Здесь ответ

Ссылки:
 - Список ссылок на параграфы из базы знаний с цитатами (максимально 200 символов)

"""

prompt = PromptTemplate(
  template=template,
  input_variables=["context", "question"]
)

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

rag_chain = (
  {"context": RunnablePassthrough(),  "question": RunnablePassthrough()}
  | prompt
  | llm
  | StrOutputParser()
)

# Initialize Flask app
app = Flask(__name__)

# Define a route to create a new knowledge base item
@app.route('/create_kb_item', methods=['POST'])
def create_kb_item():
    event = request.get_json()["event"]
    document_id = event["data"]["new"]["id"]

    if "document" in event["data"]["new"]["message_data"]:
        document = event["data"]["new"]["message_data"]["document"]

        if document["mime_type"] == "application/pdf":
            response = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getFile?file_id={document['file_id']}")
            response.raise_for_status()

            file_info = response.json()

            print(file_info)
            file_content_response = requests.get(f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info['result']['file_path']}")
            file_content_response.raise_for_status()

            from PyPDF2 import PdfReader
            import io

            # Now, creating a PdfReader object from the downloaded content
            pdf_reader = PdfReader(
                stream=io.BytesIO(file_content_response.content)  # Create stream object from the response content
            )

            # Initialize an empty string to store the extracted text
            extracted_text = ""

            # Loop through all the pages in the PDF file
            for page_num in range(len(pdf_reader.pages)):
                # Extract the text from the current page
                page_text = pdf_reader.pages[page_num].extract_text()

                # Add the extracted text to the final text
                extracted_text += page_text

            kb.add_document(document_id, extracted_text)

        print(document["mime_type"])


    if "text" in event["data"]["new"]["message_data"]:
        document_text = event["data"]["new"]["message_data"]["text"]

        if len(document_text) < 140:
            return jsonify("Skipped"), 200

        kb.add_document(document_id, document_text)

    return jsonify('Ok'), 200


@app.route('/get_question_response', methods=['POST'])
def get_question_response():
    question = request.get_json()["input"]["question"]

    response = rag_chain.invoke({
        "context": kb.query(question),
        "question": question,
    })

    print(response)

    return jsonify(response), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

