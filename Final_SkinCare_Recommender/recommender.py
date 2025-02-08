import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SkinCareRecommender:
    def __init__(self, df):
        self.df = df.copy()
        self.preprocess_data()

    def preprocess_data(self):
        """Preprocess the dataset by combining key features and standardizing text."""
        self.df['Ingredients'] = self.df['Ingredients'].str.replace(',', ' ')
        self.df['Feature_Blob'] = (
            self.df['Skin Concern'] + ' ' + self.df['Severity'] + ' ' +
            self.df['Ingredients'] + ' ' + self.df['Product Type']
        )
        self.df['Skin Type'] = self.df['Skin Type'].str.lower()

    def get_recommendations(self, skin_concern, severity, price_range=(0, 1000), top_n=5):
        """Get product recommendations based on user preferences."""
        # Convert skin_concern and severity to lowercase
        skin_concern = [concern.lower() for concern in skin_concern]
        severity = severity.lower()

        # Filter by price range
        filtered_df = self.df[(self.df['Price'] >= price_range[0]) & (self.df['Price'] <= price_range[1])]
        if filtered_df.empty:
            return pd.DataFrame()

        # Ensure exact skin concern match (case-insensitive)
        concern_df = filtered_df[filtered_df['Skin Concern'].apply(
            lambda x: any(concern in x.lower().split(',') for concern in skin_concern)
        )]
        if concern_df.empty:
            return pd.DataFrame()

        # Filter by severity (case-insensitive)
        severity_df = concern_df[concern_df['Severity'].str.contains(severity, case=False, na=False)]
        if severity_df.empty:
            return pd.DataFrame()

        # Create user query (in lowercase)
        user_query = ' '.join(skin_concern + [severity])
        product_features = severity_df['Feature_Blob']

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([user_query] + product_features.tolist())

        # Compute cosine similarity
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Create a copy of severity_df to avoid modifying the original DataFrame directly
        severity_df_copy = severity_df.copy()
        severity_df_copy['Score'] = cosine_similarities

        # Sort by similarity score and customer rating
        sorted_df = severity_df_copy.sort_values(by=['Score', 'Customer Rating'], ascending=[False, False])
        
        # Reset the index and assign custom index starting from 1
        sorted_df = sorted_df.reset_index(drop=True)
        sorted_df.index = sorted_df.index + 1  # Set custom index starting from 1
        
        # Return top_n products with relevant details
        return sorted_df.head(top_n)[['Product Name', 'Brand', 'Ingredients', 'Price', 'Customer Rating', 'Availability', 'Score']]
    
    def calculate_severity(self, detected_issues):
        total_severity = 0
        total_count = 0
    
        base_severity_map = {
            "blackheads": 1,
            "papules": 2,
            "nodules": 3,
            "pustules": 4,
            "dark spots": 2,
            "whiteheads": 1,
        }
    
        for issue, count in detected_issues.items():
            base_severity = base_severity_map.get(issue, 0)
            adjusted_severity = base_severity + (count - 1) * 0.5
            total_severity += adjusted_severity * count
            total_count += count
    
        overall_score = total_severity / total_count if total_count > 0 else 0
    
        if overall_score >= 3:
            return "High", overall_score
        elif overall_score >= 2:
            return "Medium", overall_score
        else:
            return "Low", overall_score

# Example usage
if __name__ == "__main__":

    # Load the dataset
    df = pd.read_csv(r'C:\Users\Samiksha Bhatia\Acne_gpu\myvenv\SkinCare_Recommendation_Final\skincare_products_1500_unique.csv')

    recommender = SkinCareRecommender(df)

    user_skin_type = "dry"  # Use lowercase skin type
    user_concern = ["nodules", "blackheads", "darkspot"]  # Use lowercase skin concerns
    user_severity = "low"  # Use lowercase severity
    max_price = 1000

    recommendations = recommender.get_recommendations(
        skin_concern=user_concern,
        severity=user_severity,
        price_range=(0, max_price),
        top_n=5
    )

    if not recommendations.empty:
        print(f"\nRecommended products for skin with {', '.join(user_concern)} concerns and {user_severity} severity:\n")
        print(recommendations.to_string(index=True))  # Index starts from 1
    else: 
        print("No products found matching your criteria. Please try different parameters.")
