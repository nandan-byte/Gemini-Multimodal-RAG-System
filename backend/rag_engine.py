import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import base64, io, os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# =========================================================
# Load environment and configure Gemini
# =========================================================
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# =========================================================
# Initialize CLIP model for multimodal embeddings
# =========================================================
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()


# =========================================================
# Embedding Functions
# =========================================================
def embed_image(image_data):
    """Generate image embedding using CLIP."""
    if isinstance(image_data, str):
        image = Image.open(image_data).convert("RGB")
    else:
        image = image_data
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()


def embed_text(text):
    """Generate text embedding using CLIP."""
    inputs = clip_processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()


# =========================================================
# PDF Processing
# =========================================================
def process_pdf(pdf_path):
    """Extract text and images from a PDF and embed both."""
    doc = fitz.open(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_docs, all_embeddings, image_data_store = [], [], {}

    for i, page in enumerate(doc):
        # -------- Extract text --------
        text = page.get_text()
        if text.strip():
            temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
            chunks = splitter.split_documents([temp_doc])
            for chunk in chunks:
                all_docs.append(chunk)
                all_embeddings.append(embed_text(chunk.page_content))

        # -------- Extract images --------
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                image_id = f"page_{i}_img_{img_index}"
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_data_store[image_id] = img_base64

                emb = embed_image(pil_image)
                all_embeddings.append(emb)
                all_docs.append(
                    Document(
                        page_content=f"[Image: {image_id}]",
                        metadata={"page": i, "type": "image", "image_id": image_id},
                    )
                )
            except Exception as e:
                print(f"⚠️ Error on page {i}: {e}")
                continue

    doc.close()

    embeddings_array = np.array(all_embeddings)
    vector_store = FAISS.from_embeddings(
        text_embeddings=[(d.page_content, e) for d, e in zip(all_docs, embeddings_array)],
        embedding=None,
        metadatas=[d.metadata for d in all_docs],
    )
    return vector_store, all_docs, image_data_store


# =========================================================
# Retrieval
# =========================================================
def retrieve_multimodal(query, vector_store, k=5):
    query_embedding = embed_text(query)
    results = vector_store.similarity_search_by_vector(query_embedding, k=k)
    return results


# =========================================================
# Gemini Model Auto-Selection
# =========================================================
def get_gemini_model():
    """Auto-selects the latest supported Gemini model."""
    preferred_models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]
    try:
        available = [m.name for m in genai.list_models()]
        for model_name in preferred_models:
            if any(model_name in a for a in available):
                print(f"✅ Using Gemini model: {model_name}")
                return genai.GenerativeModel(model_name)
        # Fallback if not found
        print("⚠️ Preferred models not found, using gemini-pro as fallback.")
        return genai.GenerativeModel("gemini-pro")
    except Exception as e:
        print("⚠️ Could not list models, defaulting to gemini-2.5-flash:", e)
        return genai.GenerativeModel("gemini-2.5-flash")


# =========================================================
# Gemini Multimodal Answer
# =========================================================
def gemini_multimodal_answer(query, retrieved_docs, image_data_store):
    text_context = ""
    images = []

    for doc in retrieved_docs:
        if doc.metadata.get("type") == "text":
            text_context += f"\n[Page {doc.metadata['page']}] {doc.page_content}"
        elif doc.metadata.get("type") == "image":
            img_id = doc.metadata["image_id"]
            if img_id in image_data_store:
                img_bytes = base64.b64decode(image_data_store[img_id])
                images.append(Image.open(io.BytesIO(img_bytes)))

    prompt = f"""
    You are an expert AI assistant.
    Use the given text and images to answer accurately.
    Question: {query}
    Context: {text_context}
    """

    model = get_gemini_model()
    response = model.generate_content([prompt] + images)
    return response.text


# =========================================================
# Main RAG Pipeline
# =========================================================
def multimodal_pdf_rag_pipeline(query, pdf_path):
    """Main Multimodal RAG pipeline."""
    vector_store, all_docs, image_data_store = process_pdf(pdf_path)
    retrieved_docs = retrieve_multimodal(query, vector_store, k=5)
    answer = gemini_multimodal_answer(query, retrieved_docs, image_data_store)
    return answer
