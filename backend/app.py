from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_engine import multimodal_pdf_rag_pipeline
import os

app = Flask(__name__)
CORS(app)  # Enable cross-origin for frontend

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF uploaded"}), 400

    file = request.files["pdf"]
    file_path = os.path.join(UPLOAD_FOLDER, "user_uploaded.pdf")
    file.save(file_path)
    return jsonify({"message": "PDF uploaded successfully", "path": file_path})


@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    query = data.get("query")
    pdf_path = os.path.join(UPLOAD_FOLDER, "user_uploaded.pdf")

    if not os.path.exists(pdf_path):
        return jsonify({"error": "No PDF uploaded yet!"}), 400
    if not query:
        return jsonify({"error": "No question provided!"}), 400

    try:
        answer = multimodal_pdf_rag_pipeline(query, pdf_path)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=8000, debug=True)
