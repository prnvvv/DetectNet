import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Network Intrusion Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set relative paths for data and model
# Using relative paths and Path for cross-platform compatibility
MODEL_PATH = Path("Network_Intrusion_Detection.pkl")
DATA_PATH = Path("Network_Intrusion_Detection_Dataset.csv")

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

# Enhanced dark theme styling
st.markdown("""
    <style>
    /* Main app background */
    .stApp {
        background-color: #121212;
        color: #ffffff;
    }
    
    /* Headers with subtle accent */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        border-bottom: 1px solid #444444;
        padding-bottom: 8px;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1wrcr25 {
        background-color: #1a1a1a !important;
        border-right: 1px solid #333333 !important;
    }
    
    /* Cards with subtle glow */
    .css-1y4p8pa, .css-1xarl3l {
        background-color: #1e1e1e;
        border: 1px solid #333333;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* Input fields */
    .stNumberInput > div > div > input {
        color: #ffffff !important;
        background-color: #2a2a2a !important;
        border: 1px solid #444444 !important;
    }
    
    /* Buttons with hover effect */
    .stButton > button {
        background-color: #2a2a2a;
        color: #ffffff;
        border: 1px solid #555555;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #3a3a3a;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* Tabs with active indicator */
    .stTabs [data-baseweb="tab"] {
        color: #aaaaaa !important;
        padding: 8px 16px !important;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        font-weight: bold;
        border-bottom: 2px solid #4a8fe7 !important;
    }
    
    /* Alerts with better contrast */
    .stAlert {
        border-left: 4px solid !important;
    }
    .stSuccess {
        border-left-color: #4CAF50 !important;
        background-color: #1B5E20 !important;
    }
    .stError {
        border-left-color: #F44336 !important;
        background-color: #B71C1C !important;
    }
    .stInfo {
        border-left-color: #2196F3 !important;
        background-color: #0D47A1 !important;
    }
    
    /* Plot containers */
    .js-plotly-plot, .plotly, .plotly-container {
        background-color: #1e1e1e !important;
        border-radius: 8px;
        border: 1px solid #333333;
    }
    
    /* Footer styling */
    footer {
        color: #666666 !important;
        font-size: 0.9em;
        text-align: center;
        padding: 16px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None

# Load functions with better error handling
@st.cache_resource
def load_model():
    try:
        model_path = MODEL_PATH
        if not model_path.exists():
            st.warning(f"Model file not found at {model_path}. Using demo mode.")
            # Create a simple demo model
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            if st.session_state.sample_data is not None:
                X = st.session_state.sample_data[FEATURES]
                y = st.session_state.sample_data['Label']
                model.fit(X, y)
            return model
        return pickle.load(open(model_path, "rb"))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_data():
    try:
        data_path = DATA_PATH
        if not data_path.exists():
            st.warning(f"Data file not found at {data_path}. Using generated sample data.")
            # Generate sample data
            np.random.seed(42)
            sample_size = 1000
            data = {}
            
            # Generate sample features
            for feature in FEATURES:
                data[feature] = np.random.normal(100, 30, sample_size)
            
            # Generate sample labels (0-5)
            data['Label'] = np.random.randint(0, 6, sample_size)
            
            df = pd.DataFrame(data)
            st.session_state.sample_data = df
            return df
            
        return pd.read_csv(data_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Create minimal sample data
        df = pd.DataFrame({feature: [100] for feature in FEATURES})
        df['Label'] = [0]
        return df

# Initialize
df = load_data()
if df is not None:
    st.session_state.data_loaded = True
    model = load_model()
    if model is not None:
        st.session_state.model_loaded = True

# Attack types mapping
attack_types = {
    0: "LEGITIMATE NETWORK TRAFFIC",
    1: "DDoS ATTACK DETECTED",
    2: "PROTOCOL EXPLOITATION DETECTED",
    3: "RECONNAISSANCE DETECTED",
    4: "TRAFFIC MANIPULATION DETECTED",
    5: "BUFFER OVERFLOW DETECTED"
}

# Title section with improved layout
col1, col2 = st.columns([1, 3])
with col1:
    # Use emoji instead of external URL
    st.markdown("# 🛡️", unsafe_allow_html=True)
with col2:
    st.title("Network Intrusion Detection System")
    st.markdown("""
        <div style='background-color: #1e1e1e; padding: 16px; border-radius: 8px; margin-bottom: 16px; border-left: 4px solid #4a8fe7;'>
            <p style='color: #ffffff; margin: 0;'>AI-powered network security monitoring and threat detection</p>
        </div>
    """, unsafe_allow_html=True)

# Sidebar with improved organization
with st.sidebar:
    # Use emoji instead of external URL
    st.markdown("# 🌐", unsafe_allow_html=True)
    st.header("Navigation")
    page = st.selectbox("Select page", ["Attack Detection", "Analytics", "Detection Result"])
    
    st.markdown("---")
    st.markdown("### System Overview")
    
    # Status indicators
    if st.session_state.model_loaded:
        st.success("Model: Loaded")
    else:
        st.error("Model: Not Loaded")
    
    if st.session_state.data_loaded:
        st.success("Data: Loaded")
    else:
        st.error("Data: Not Loaded")
    
    st.metric("Total Records", len(df), help="Number of network traffic records in dataset")
    st.metric("Attack Types", len(attack_types), help="Different types of attacks detected")
    
    # Add reset button
    if st.button("Reset Detection"):
        st.session_state.prediction = None
        st.success("Detection results reset!")

# Plotly theme configuration
plotly_layout = dict(
    paper_bgcolor="#1e1e1e",
    plot_bgcolor="#1e1e1e",
    font=dict(color="#ffffff"),
    title_font_color="#ffffff",
    legend_font_color="#ffffff",
    xaxis=dict(gridcolor="#333333", zerolinecolor="#333333"),
    yaxis=dict(gridcolor="#333333", zerolinecolor="#333333"),
    margin=dict(l=20, r=20, t=40, b=20)
)

# Function to get default values for inputs
def get_default(feature):
    try:
        return float(df[feature].mean())
    except:
        return 100.0  # Fallback value

if page == "Attack Detection":
    st.header("Network Attack Detection")
    
    # Instruction card
    with st.expander("ℹ️ How to use this tool", expanded=True):
        st.markdown("""
            1. Enter network traffic parameters below
            2. Click 'Detect Attack' to analyze
            3. View results in the Detection Result page
        """)
    
    # Quick fill buttons for easy testing
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Fill with Average Values"):
            for feature in FEATURES:
                st.session_state[f"input_{feature}"] = get_default(feature)
            st.success("Form filled with average values")
    
    with col2:
        if st.button("Fill with Sample DDoS Attack"):
            # Set values typical for DDoS attack
            for feature in FEATURES:
                # Set high values for traffic-related features
                if "Packets" in feature or "Bytes" in feature:
                    st.session_state[f"input_{feature}"] = get_default(feature) * 5
                else:
                    st.session_state[f"input_{feature}"] = get_default(feature)
            st.success("Form filled with DDoS attack pattern")
    
    with col3:
        if st.button("Clear Form"):
            for feature in FEATURES:
                st.session_state[f"input_{feature}"] = get_default(feature)
            st.success("Form cleared")
    
    # Input form with better organization
    st.subheader("Network Traffic Parameters")
    cols = st.columns(3)
    input_data = {}
    
    for i, feature in enumerate(FEATURES):
        with cols[i % 3]:
            input_data[feature] = st.number_input(
                label=feature,
                value=get_default(feature),
                help=f"Average value: {get_default(feature):.2f}",
                key=f"input_{feature}"
            )
    
    # Enhanced detection button
    detect_col1, detect_col2 = st.columns([3, 1])
    with detect_col1:
        if st.button("🔍 Detect Attack", type="primary", use_container_width=True):
            if not st.session_state.model_loaded:
                st.error("Model not loaded properly. Please check the model file.")
            else:
                with st.spinner('Analyzing network traffic patterns...'):
                    try:
                        input_df = pd.DataFrame([input_data])[FEATURES]
                        st.session_state.prediction = model.predict(input_df)[0]
                        st.success("Analysis complete! Switch to the Detection Result page to view findings.")
                    except Exception as e:
                        st.error(f"Error in attack detection: {str(e)}")
                        st.info("Try resetting input values or check if all required features are provided.")
    
    with detect_col2:
        st.write("")
        st.write("")
        if st.button("Reset", use_container_width=True):
            st.session_state.prediction = None
            st.success("Detection reset")

elif page == "Analytics":
    st.header("Network Analytics")
    
    if not st.session_state.data_loaded:
        st.error("Data not loaded properly. Analytics unavailable.")
    else:
        # Simplified tabs
        tab1, tab2 = st.tabs(["Attack Distribution", "Feature Correlation"])
        
        with tab1:
            st.subheader("Attack Type Distribution")
            
            # Create attack type column with labels
            attack_labels = df['Label'].map(attack_types)
            
            # Enhanced pie chart
            try:
                fig = px.pie(
                    df,
                    names=attack_labels,
                    hole=0.3,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig.update_layout(plotly_layout)
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.markdown("#### Summary Statistics")
                col1, col2 = st.columns(2)
                
                # Find most common attack - handle errors gracefully
                try:
                    most_common = attack_types[df['Label'].mode()[0]]
                except:
                    most_common = "Unknown"
                
                # Calculate attack rate safely
                try:
                    attack_rate = f"{100*(df['Label'] != 0).mean():.1f}%"
                except:
                    attack_rate = "N/A"
                
                col1.metric("Most Common Attack", most_common)
                col2.metric("Attack Rate", attack_rate)
            except Exception as e:
                st.error(f"Error generating attack distribution chart: {str(e)}")
                st.info("This may be due to missing or invalid data.")
        
        with tab2:
            st.subheader("Feature Correlation Analysis")
            
            try:
                # Correlation heatmap with controls
                num_features = st.slider(
                    "Number of top features to display",
                    min_value=5,
                    max_value=min(15, len(FEATURES)),
                    value=min(10, len(FEATURES)),
                    help="Show the most relevant features for attack detection"
                )
                
                # Handle case where Label might not exist
                if 'Label' in df.columns:
                    correlations = df.corr()['Label'].abs().sort_values(ascending=False)
                    top_features = correlations.head(num_features).index
                    
                    corr_matrix = df[top_features].corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        color_continuous_scale='RdBu_r',
                        text_auto=".2f",
                        aspect="auto"
                    )
                    fig.update_layout(plotly_layout)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Label column not found in dataset. Cannot compute feature correlations with attack labels.")
                    
                    # Show general correlation matrix instead
                    corr_matrix = df[FEATURES[:num_features]].corr()
                    fig = px.imshow(
                        corr_matrix,
                        color_continuous_scale='RdBu_r',
                        text_auto=".2f",
                        aspect="auto"
                    )
                    fig.update_layout(plotly_layout)
                    st.plotly_chart(fig, use_container_width=True)
                    st.info("Showing general feature correlations instead.")
            except Exception as e:
                st.error(f"Error generating correlation analysis: {str(e)}")
                st.info("This may be due to missing data or invalid feature values.")

else:  # Detection Result page
    st.header("Detection Result")
    
    if st.session_state.prediction is not None:
        result = attack_types.get(st.session_state.prediction, "UNKNOWN TRAFFIC PATTERN")
        is_legitimate = st.session_state.prediction == 0
        
        # Enhanced result display
        if is_legitimate:
            st.success(f"### ✅ {result}")
            st.markdown("""
                <div style='background-color: #1e1e1e; padding: 16px; border-radius: 8px; border-left: 4px solid #4CAF50; margin: 16px 0;'>
                    <p style='color: #ffffff; margin: 0;'>No malicious activity detected in the analyzed network traffic.</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Additional details for legitimate traffic
            with st.expander("📊 Traffic Analysis Details", expanded=True):
                st.markdown("""
                * Traffic patterns align with normal network behavior
                * No anomalous connection attempts detected
                * Packet rates within expected thresholds
                * Protocol usage consistent with legitimate applications
                """)
        else:
            st.error(f"### ⚠️ {result}")
            st.markdown("""
                <div style='background-color: #1e1e1e; padding: 16px; border-radius: 8px; border-left: 4px solid #F44336; margin: 16px 0;'>
                    <p style='color: #ffffff; margin: 0;'>Potential security threat detected in the analyzed network traffic.</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Actionable recommendations
            with st.expander("🚨 Recommended Actions", expanded=True):
                st.markdown("""
                1. **Isolate affected systems** to prevent further spread
                2. **Review security logs** for additional context
                3. **Update firewall rules** to block similar patterns
                4. **Notify security team** for further investigation
                """)
                
            # Add attack-specific information
            with st.expander("🔍 Attack Details", expanded=True):
                attack_info = {
                    1: """
                        **DDoS Attack Information:**
                        * High volume of traffic targeting specific services
                        * Multiple source IPs with similar traffic patterns
                        * Potential botnet coordination
                        * Recommend: Rate limiting, traffic filtering
                    """,
                    2: """
                        **Protocol Exploitation Information:**
                        * Unusual protocol behavior detected
                        * Potential vulnerability targeting
                        * Check for unpatched systems
                        * Recommend: Protocol validation, deep packet inspection
                    """,
                    3: """
                        **Reconnaissance Information:**
                        * Port scanning activity detected
                        * Network mapping attempts
                        * Information gathering phase of attack
                        * Recommend: Block scanning IP ranges, enhance IDS rules
                    """,
                    4: """
                        **Traffic Manipulation Information:**
                        * Evidence of packet injection or modification
                        * Potential man-in-the-middle activity
                        * Data integrity compromised
                        * Recommend: Enforce encrypted communications, validate packet integrity
                    """,
                    5: """
                        **Buffer Overflow Information:**
                        * Abnormally large packet sizes detected
                        * Potential memory corruption attempts
                        * Target service exploitation
                        * Recommend: Update vulnerable services, implement input validation
                    """
                }
                st.markdown(attack_info.get(st.session_state.prediction, "No detailed information available for this attack type."))
    else:
        st.info("""
            No detection results available.  
            Please go to the **Attack Detection** page to analyze network traffic.
        """)
        
        # Quick navigation button
        if st.button("Go to Attack Detection"):
            st.session_state.page = "Attack Detection"
            st.experimental_rerun()

# Minimal footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666666; font-size: 0.9em;'>
        <p>Network Security Dashboard • v1.0</p>
    </div>
""", unsafe_allow_html=True)