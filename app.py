import streamlit as st
import torch
import os
from model import load_trained_model, predict

# Set page config
st.set_page_config(
    page_title="DeepFake Audio Detection",
    page_icon="🎵",
    layout="centered"
)

# Add custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #f5f5f5;
}
.uploadedFile {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
.prediction {
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎵 DeepFake Audio Detection")
st.markdown("""
    Upload an audio file to check if it's authentic or artificially generated.
    The model will analyze the audio and provide a prediction.
""")

def main():
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = load_trained_model()
        model = model.to(device)
    except Exception as e:
        st.error("Error loading the model. Please make sure the model file exists and is valid.")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file", 
        type=['wav', 'mp3'],
        help="Upload a WAV or MP3 file (max 200MB) for analysis"
    )

    if uploaded_file is not None:
        # Check file size (200MB limit)
        file_size = len(uploaded_file.getbuffer()) / (1024 * 1024)  # Convert to MB
        if file_size > 200:
            st.error(f"File size ({file_size:.1f}MB) exceeds the 200MB limit.")
            return

        # Save the uploaded file temporarily
        with st.spinner("Processing audio..."):
            temp_path = os.path.join(os.path.dirname(__file__), "temp_audio.wav")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                # Get prediction
                prediction = predict(model, temp_path, device)
                
                # Display results
                st.subheader("Analysis Results")
                
                # Create a progress bar for the prediction
                prob_fake = prediction * 100
                prob_real = (1 - prediction) * 100
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="Probability of being Real",
                        value=f"{prob_real:.1f}%"
                    )
                
                with col2:
                    st.metric(
                        label="Probability of being Fake",
                        value=f"{prob_fake:.1f}%"
                    )

                # Final verdict
                if prediction > 0.5:
                    st.error("🚨 This audio is likely to be FAKE")
                else:
                    st.success("✅ This audio is likely to be AUTHENTIC")

            except Exception as e:
                st.error(f"Error processing the audio: {str(e)}")
            finally:
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)

if __name__ == "__main__":
    main()