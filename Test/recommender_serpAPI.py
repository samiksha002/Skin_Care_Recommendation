import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load embedding model
embedding_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Knowledge base of skincare ingredients
knowledge_base = [
    "Salicylic acid. A beta hydroxy acid that helps with acne and exfoliates the skin.",
    "Glycolic acid. An alpha hydroxy acid that exfoliates and improves skin texture.",
    "Vitamin C. Known for brightening the skin and reducing hyperpigmentation.",
    "Hyaluronic acid. A humectant that hydrates and plumps the skin.",
]

# Convert knowledge base into embeddings
knowledge_embeddings = embedding_model.encode(knowledge_base)

def retrieve_ingredients(user_query, top_n=3):
    query_embedding = embedding_model.encode([user_query])
    similarities = cosine_similarity(query_embedding, knowledge_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    ingredients_list = [knowledge_base[i].split(".")[0] for i in top_indices]
    return ingredients_list

def get_real_time_products(ingredients):
    api_key = "13df0b8746edf3f222f05b575e1f6960bb231a2c4c2cff08bd908afd761acd26"  
    query = f"best skincare products for acne containing {', '.join(ingredients)}"
    
    url = f"https://serpapi.com/search.json?q={query}&tbm=shop&api_key={api_key}"
    
    response = requests.get(url)
    data = response.json()
    
    if "shopping_results" in data:
        products = []
        for item in data["shopping_results"][:5]:  # Get top 5 results
            product_name = item.get("title", "No title available")
            price = item.get("price", "Price not available")
            link = item.get("link", "#")
            products.append(f"{product_name} - {price} [Buy here]({link})")
        
        return products
    else:
        return ["No real-time products found."]

# Example query
user_query = "Best treatment for blackheads and pustules?"
retrieved_ingredients = retrieve_ingredients(user_query)
recommended_products = get_real_time_products(retrieved_ingredients)

# Print results
print("\nTop Ingredients:", retrieved_ingredients)
print("\nRecommended Products:\n", "\n".join(recommended_products))
