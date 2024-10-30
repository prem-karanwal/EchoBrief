import streamlit as st
import torch
import os
from concurrent.futures import ThreadPoolExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline
from pydub import AudioSegment

os.environ['GOOGLE_API_KEY'] = ''
api_key = os.getenv('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(
    model="gemini-pro", temperature=0.7
)

splitter = CharacterTextSplitter(chunk_size=2000)

device = 0 if torch.cuda.is_available() else -1  

def segment_audio(filepath: str, segment_length_ms: int = 10 * 60 * 1000):
    audio = AudioSegment.from_file(filepath)
    segments = [audio[i:i + segment_length_ms] for i in range(0, len(audio), segment_length_ms)]
    return segments

def transcribe_segment(segment_path: str):
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
        device=device  
    )
    return pipe(segment_path, batch_size=8)["text"]

def audio_transcription(filepath: str):
    try:
        file_size = os.path.getsize(filepath)
        audio = AudioSegment.from_file(filepath)
        audio_length_ms = len(audio)

        
        if file_size > 25 * 1024 * 1024 or audio_length_ms > 10 * 60 * 1000:
            segments = segment_audio(filepath)
        else:
            segments = [audio]

        transcriptions = []
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for i, segment in enumerate(segments):
                segment_path = f"temp_segment_{i}.mp3"
                segment.export(segment_path, format="mp3")
                
                futures.append(executor.submit(transcribe_segment, segment_path))
            
           
            for future in futures:
                transcriptions.append(future.result())

        combined_transcription = " ".join(transcriptions)
        print(combined_transcription)
        chunks = splitter.split_text(combined_transcription)
        summary = ""

        
        for chunk in chunks:
            response = llm.invoke(input=f"Summarize {input}: \n\n{chunk}")
            summary += response.content.strip() + "\n\n"

        return summary

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return str(e)

st.title("EchoBrief: Streamlining Your Audio Content")
input = st.text_input("Could you describe the content of the audio, such as whether it’s a meeting, a story, or something else?")
st.write("Upload an audio file to get the summarized text.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    temp_file_path = "temp_audio_file"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    
    summary = audio_transcription(temp_file_path)
    
    st.write("Summary:")
    st.write(summary)
    
def add_footer():
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            color: black;
            text-align: center;
            padding: 10px;
            font-size: small;
        }
        </style>
        <div class="footer">
            <p>© Made By Prem Karanwal</p>
        </div>
        """,
        unsafe_allow_html=True
    )

add_footer()