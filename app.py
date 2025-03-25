from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os
import json
import uuid
import time
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime

app = Flask(__name__)
openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Pinecone setup
pinecone_api_key = os.environ['PINECONE_API_KEY']
index_name = 'actionmemory'

pc = Pinecone(api_key=pinecone_api_key)
spec = ServerlessSpec(cloud='aws', region='us-west-2')

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
  pc.create_index(name=index_name, dimension=1536, metric='cosine', spec=spec)

while not pc.describe_index(index_name).status['ready']:
  time.sleep(1)

index = pc.Index(index_name)


@app.route('/add_observation', methods=['POST'])
def add_observation():
  data = request.json
  if not data or 'text' not in data:
    return jsonify({"error": "No observation text provided"}), 400

  metadata = {
      "text": data['text'],
      "type": data.get('type', 'observation'),  # observation, theory, symptom
      "timestamp": datetime.utcnow().isoformat(),
      "tags": data.get('tags', []),
      "category": data.get('category',
                           'general')  # diet, skin, medication, etc
  }

  embedding_vector = get_embedding_vector(data['text'])
  if embedding_vector is None:
    return jsonify({"error": "Failed to process observation"}), 500

  success = save_to_pinecone(metadata, embedding_vector)
  if not success:
    return jsonify({"error": "Failed to save observation"}), 500

  return jsonify({
      "message": "Observation added successfully",
      "metadata": metadata
  }), 200


@app.route('/search', methods=['GET'])
def search():
  text = request.args.get('text')
  category = request.args.get('category')
  type = request.args.get('type')

  if not text:
    return jsonify({"error": "No search text provided"}), 400

  embedding_vector = get_embedding_vector(text)
  if embedding_vector is None:
    return jsonify({"error": "Failed to process search"}), 500

  similar_items = query_similar_texts(embedding_vector, category, type)
  return jsonify(similar_items)


@app.route('/delete', methods=['POST'])
def delete():
  data = request.json
  if not data or 'id' not in data:
    return jsonify({"error": "No observation ID provided"}), 400

  try:
    index.delete(ids=[data['id']])
    return jsonify({"message": "Observation deleted successfully"}), 200
  except Exception as e:
    return jsonify({"error": f"Failed to delete observation: {str(e)}"}), 500


def get_embedding_vector(text):
  try:
    response = openai_client.embeddings.create(input=[text],
                                               model="text-embedding-3-large")
    return response.data[0].embedding
  except Exception as e:
    print(f"Error generating embeddings: {e}")
    return None


def query_similar_texts(embedding_vector, category=None, type=None):
  filter = {}
  if category:
    filter["category"] = category
  if type:
    filter["type"] = type

  query_results = index.query(vector=embedding_vector,
                              top_k=10,
                              include_metadata=True,
                              filter=filter)

  return [{
      'id': match['id'],
      'score': match['score'],
      'metadata': match.get('metadata', {})
  } for match in query_results['matches']]


def save_to_pinecone(metadata, embedding_vector):
  try:
    unique_id = str(uuid.uuid4())
    index.upsert(vectors=[(unique_id, embedding_vector, metadata)])
    return True
  except Exception as e:
    print(f"Error saving to Pinecone: {e}")
    return False


if __name__ == '__main__':
  app.run(debug=True, port=80, host='0.0.0.0')
