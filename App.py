import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set absolute paths
MODEL_PATH = r"C:\Users\venka\OneDrive\Dokumen\GitHub\DetectNet\Network_Intrusion_Detection.pkl"
DATA_PATH = r"C:\Users\venka\OneDrive\Dokumen\GitHub\DetectNet\Network_Intrusion_Detection_Dataset.csv"

# Define the exact features used in training
FEATURES = [
    'Port Number', 'Received Packets', 'Received Bytes', 'Sent Bytes', 
    'Sent Packets', 'Port alive Duration (S)', 'Packets Rx Dropped',
    'Packets Tx Dropped', 'Packets Rx Errors', 'Packets Tx Errors',
    'Delta Received Packets', 'Delta Received Bytes', 'Delta Sent Bytes',
    'Delta Sent Packets', 'Delta Port alive Duration (S)',
    'Delta Packets Rx Dropped', ' Delta Packets Tx Dropped',
    'Delta Packets Rx Errors', 'Delta Packets Tx Errors',
    'Connection Point', 'Total Load/Rate', 'Total Load/Latest',
    'Unknown Load/Rate', 'Unknown Load/Latest', 'Latest bytes counter',
    'is_valid', 'Table ID', 'Active Flow Entries', 'Packets Looked Up',
    'Packets Matched', 'Max Size'
]

# Set page config
st.set_page_config(page_title="Network Intrusion Detection", layout="wide")

# Initialize session state for storing prediction
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Load the model
@st.cache_resource
def load_model():
    try:
        return pickle.load(open(MODEL_PATH, "rb"))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the dataset
@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Initialize
model = load_model()
df = load_data()

if model is None or df is None:
    st.error("Failed to load model or data. Please check the file paths.")
    st.stop()

# Title
st.title("Network Intrusion Detection System")

# Define attack types mapping
attack_types = {
    0: "🟢 LEGITIMATE NETWORK TRAFFIC",
    1: "🔴 DDoS ATTACK DETECTED",
    2: "🔴 PROTOCOL EXPLOITATION DETECTED",
    3: "🔴 RECONNAISSANCE DETECTED",
    4: "🔴 TRAFFIC MANIPULATION DETECTED",
    5: "🔴 BUFFER OVERFLOW DETECTED"
}

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Attack Detection", "Analysis Graphs", "Detection Result"])

if page == "Attack Detection":
    st.header("Network Attack Detection")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    input_data = {}
    # Create input fields dynamically
    for i, feature in enumerate(FEATURES):
        with col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3:
            input_data[feature] = st.number_input(
                f"{feature}",
                value=float(df[feature].mean()),
                help=f"Average value: {df[feature].mean():.2f}"
            )
    
    # Make prediction button
    if st.button("Detect Attack"):
        # Prepare input data
        input_df = pd.DataFrame([input_data])[FEATURES]
        
        try:
            # Make prediction and store in session state
            st.session_state.prediction = model.predict(input_df)[0]
            st.success("Detection complete! View result in the 'Detection Result' page.")
            
        except Exception as e:
            st.error(f"Error in attack detection: {e}")

elif page == "Analysis Graphs":
    st.header("Network Traffic Analysis")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Attack Distribution", "Correlation Analysis", "Feature Analysis"])
    
    with tab1:
        st.subheader("Distribution of Attack Types")
        fig, ax = plt.subplots(figsize=(12, 6))
        attack_dist = df['Label'].value_counts().sort_index()
        attack_names = list(attack_types.values())
        colors = ['green' if i == 0 else 'red' for i in range(len(attack_names))]
        ax.bar(attack_names, attack_dist.values, color=colors)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.subheader("Feature Correlation Analysis")
        num_features = st.slider("Number of features to show", 5, 15, 10)
        correlations = df.corr()['Label'].abs().sort_values(ascending=False)
        top_features = correlations.head(num_features).index
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df[top_features].corr(), 
                    cmap='coolwarm', 
                    center=0, 
                    annot=True, 
                    fmt='.2f')
        plt.title(f"Top {num_features} Most Correlated Features")
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.subheader("Feature Distribution Analysis")
        selected_feature = st.selectbox("Select feature to analyze", FEATURES)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        sns.histplot(data=df, x=selected_feature, hue='Label', ax=ax1)
        ax1.set_title(f"Distribution of {selected_feature}")
        
        sns.boxplot(data=df, x='Label', y=selected_feature, ax=ax2)
        ax2.set_title(f"Box Plot of {selected_feature} by Attack Type")
        
        plt.tight_layout()
        st.pyplot(fig)

else:  # Detection Result page
    st.header("Detection Result")
    
    if st.session_state.prediction is not None:
        # Display result in large, centered format
        st.markdown(
            f"""
            <div style='text-align: center; padding: 50px; background-color: {'black' if st.session_state.prediction == 0 else 'black'}; 
                        border-radius: 10px; margin: 20px 0;'>
                <h1 style='font-size: 2.5em;'>{attack_types[st.session_state.prediction]}</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.info("No detection results yet. Please go to the 'Attack Detection' page to analyze network traffic.")

# Add footer
st.markdown("---")
st.markdown("Network Intrusion Detection System - Powered by Machine Learning")