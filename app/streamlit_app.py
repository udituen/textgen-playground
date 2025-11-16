# streamlit_app.py
import streamlit as st
import requests
import json
from typing import Optional

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

# Page config
st.set_page_config(
    page_title="Text Generator",
    page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("Text Generator Play ground")
st.markdown("Generate text using various language models and decoding strategies")

# Sidebar for model configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    model_name = st.text_input(
        "Model Name",
        value="gpt2",
        help="Enter HuggingFace model name (e.g., 'gpt2', 'Qwen/Qwen-7B-Chat')"
    )
    
    # Decoding strategy
    strategy = st.selectbox(
        "Decoding Strategy",
        options=["greedy", "top_k", "top_p", "beam"],
        index=0,
        help="Choose the decoding strategy for text generation"
    )
    
    # Temperature
    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Higher values make output more random, lower values more deterministic"
    )
    
    # Max new tokens
    max_new_tokens = st.slider(
        "Max New Tokens",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Maximum number of tokens to generate"
    )
    
    # Strategy-specific parameters
    st.markdown("---")
    st.subheader("Strategy Parameters")
    
    top_k = None
    top_p = None
    num_beams = None
    
    if strategy == "top_k":
        top_k = st.slider(
            "Top K",
            min_value=1,
            max_value=100,
            value=50,
            help="Number of highest probability tokens to keep"
        )
    
    elif strategy == "top_p":
        top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Cumulative probability threshold for nucleus sampling"
        )
    
    elif strategy == "beam":
        num_beams = st.slider(
            "Num Beams",
            min_value=1,
            max_value=10,
            value=4,
            help="Number of beams for beam search"
        )
    
    # Model management
    st.markdown("---")
    st.subheader("Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Load Model", use_container_width=True):
            with st.spinner(f"Loading {model_name}..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/models/load/{model_name}"
                    )
                    if response.status_code == 200:
                        st.success("Model loaded!")
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
    
    with col2:
        if st.button("Clear Cache", use_container_width=True):
            try:
                response = requests.post(f"{API_BASE_URL}/models/clear-cache")
                if response.status_code == 200:
                    st.success("Cache cleared!")
                else:
                    st.error("Failed to clear cache")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input")
    
    # Prompt input
    prompt = st.text_area(
        "Enter your prompt:",
        height=200,
        placeholder="Once upon a time...",
        help="Enter the text prompt for generation"
    )
    
    # Generate button
    generate_button = st.button(
        "Generate",
        type="primary",
        use_container_width=True
    )

with col2:
    st.header("Output")
    
    # Placeholder for output
    output_container = st.empty()

# Generation logic
if generate_button:
    if not prompt:
        st.warning("Please enter a prompt!")
    else:
        # Prepare request
        request_data = {
            "prompt": prompt,
            "model_name": model_name,
            "strategy": strategy,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        }
        
        # Add strategy-specific parameters
        if top_k is not None:
            request_data["top_k"] = top_k
        if top_p is not None:
            request_data["top_p"] = top_p
        if num_beams is not None:
            request_data["num_beams"] = num_beams
        
        # Show loading spinner
        with st.spinner("Generating text..."):
            try:
                # Make API request
                response = requests.post(
                    f"{API_BASE_URL}/generate",
                    json=request_data,
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display output
                    with output_container:
                        st.text_area(
                            "Generated Text:",
                            value=result["generated_text"],
                            height=200,
                            disabled=True
                        )
                        
                        # Show metadata
                        with st.expander("Generation Details"):
                            st.json({
                                "model": result["model_name"],
                                "strategy": result["strategy"],
                                "temperature": result["temperature"],
                            })
                    
                    st.success("Generation complete!")
                    
                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    st.error(f"Error: {error_detail}")
                    
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The model might be taking too long to generate.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to API. Make sure FastAPI server is running.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Display current session info
with st.expander("ℹ About"):
    st.markdown("""
    ### Text Generator API
    
    This interface allows you to generate text using various language models and decoding strategies.
    
    **Decoding Strategies:**
    - **Greedy**: Always picks the most likely token (deterministic)
    - **Top-K**: Samples from the K most likely tokens
    - **Top-P**: Nucleus sampling - samples from smallest set with cumulative probability ≥ p
    - **Beam**: Beam search for more coherent outputs
    
    **Tips:**
    - Higher temperature = more random/creative
    - Lower temperature = more focused/deterministic
    - Greedy decoding ignores temperature
    - Pre-load models using the sidebar button for faster generation
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Built with FastAPI + Streamlit</div>",
    unsafe_allow_html=True
)