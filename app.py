
import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import tempfile
import os
import base64

st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️", layout="centered")

st.markdown("""
<style>
    .title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        background: linear-gradient(90deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle { text-align: center; color: #888; font-size: 1.1em; margin-bottom: 2em; }
    .result-fake {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        padding: 20px; border-radius: 15px; text-align: center;
        font-size: 1.8em; font-weight: bold; color: white; margin: 20px 0;
    }
    .result-real {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 20px; border-radius: 15px; text-align: center;
        font-size: 1.8em; font-weight: bold; color: white; margin: 20px 0;
    }
    .record-btn {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white; border: none; padding: 15px 40px;
        border-radius: 50px; font-size: 1.2em;
        cursor: pointer; margin: 10px;
        transition: all 0.3s;
    }
    .stop-btn {
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        color: white; border: none; padding: 15px 40px;
        border-radius: 50px; font-size: 1.2em;
        cursor: pointer; margin: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🎙️ Deepfake Audio Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect if audio is Real (Human) or Fake (AI Generated)</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", "90.74%")
col2.metric("F1 Score", "90.10%")
col3.metric("EER", "3.17%")
col4.metric("Model", "CNN")

st.divider()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5")

model = load_model()

def predict(file_path):
    audio, sr = librosa.load(file_path, sr=16000, duration=4.0)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < 200:
        mfcc = np.pad(mfcc, ((0,0),(0, 200 - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :200]
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    prob = model.predict(mfcc, verbose=0)[0][0]
    return prob

def show_result(prob):
    if prob > 0.5:
        confidence = prob * 100
        st.markdown('<div class="result-fake">🚨 FAKE — AI Generated</div>', unsafe_allow_html=True)
    else:
        confidence = (1 - prob) * 100
        st.markdown('<div class="result-real">✅ REAL — Human Voice</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Confidence", f"{confidence:.2f}%")
    c2.metric("Fake Probability", f"{prob*100:.2f}%")
    st.progress(float(prob))

tab1, tab2 = st.tabs(["📁 Upload Audio", "🎙️ Live Record"])

with tab1:
    st.markdown("### Upload an audio file")
    uploaded_file = st.file_uploader("WAV, MP3, FLAC supported", type=["wav","mp3","flac"])
    if uploaded_file:
        st.audio(uploaded_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        with st.spinner("🔍 Analyzing..."):
            prob = predict(tmp_path)
        os.unlink(tmp_path)
        show_result(prob)

with tab2:
    st.markdown("### Record directly from browser 🎙️")
    
    # HTML5 Recording Component
    recorder_html = """
    <div style="text-align:center; padding: 20px;">
        <div id="status" style="color:#888; margin-bottom:15px; font-size:1.1em;">
            Click Start to begin recording
        </div>
        <button class="record-btn" onclick="startRecording()" id="startBtn"
            style="background: linear-gradient(135deg, #f093fb, #f5576c);
            color:white; border:none; padding:15px 40px; border-radius:50px;
            font-size:1.1em; cursor:pointer; margin:5px;">
            🎙️ Start Recording
        </button>
        <button onclick="stopRecording()" id="stopBtn" disabled
            style="background: linear-gradient(135deg, #4facfe, #00f2fe);
            color:white; border:none; padding:15px 40px; border-radius:50px;
            font-size:1.1em; cursor:pointer; margin:5px; opacity:0.5;">
            ⏹️ Stop Recording
        </button>
        <br><br>
        <audio id="audioPlayback" controls style="display:none; width:100%; margin:10px 0;"></audio>
        <br>
        <a id="downloadLink" style="display:none;">
            <button style="background: linear-gradient(135deg, #11998e, #38ef7d);
                color:white; border:none; padding:12px 30px; border-radius:50px;
                font-size:1em; cursor:pointer; margin:5px;">
                ⬇️ Download Recording
            </button>
        </a>
    </div>

    <script>
    let mediaRecorder;
    let audioChunks = [];

    async function startRecording() {
        audioChunks = [];
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
        mediaRecorder.onstop = () => {
            const blob = new Blob(audioChunks, { type: "audio/wav" });
            const url = URL.createObjectURL(blob);
            
            document.getElementById("audioPlayback").src = url;
            document.getElementById("audioPlayback").style.display = "block";
            
            const link = document.getElementById("downloadLink");
            link.href = url;
            link.download = "recorded_audio.wav";
            link.style.display = "inline";
            
            document.getElementById("status").innerHTML = 
                "✅ Recording complete! Download the file and upload it above in Tab 1";
            document.getElementById("status").style.color = "#38ef7d";
        };
        
        mediaRecorder.start();
        document.getElementById("status").innerHTML = "🔴 Recording... Speak now!";
        document.getElementById("status").style.color = "#ff4b2b";
        document.getElementById("startBtn").disabled = true;
        document.getElementById("startBtn").style.opacity = "0.5";
        document.getElementById("stopBtn").disabled = false;
        document.getElementById("stopBtn").style.opacity = "1";
    }

    function stopRecording() {
        mediaRecorder.stop();
        document.getElementById("startBtn").disabled = false;
        document.getElementById("startBtn").style.opacity = "1";
        document.getElementById("stopBtn").disabled = true;
        document.getElementById("stopBtn").style.opacity = "0.5";
    }
    </script>
    """
    
    st.components.v1.html(recorder_html, height=300)
    
    st.info("👆 Record karo → Download karo → Upload Tab mein analyze karo!")
    
    recorded = st.file_uploader("Upload your recorded file here", 
                                 type=["wav","mp3","flac"], key="rec")
    if recorded:
        st.audio(recorded)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(recorded.read())
            tmp_path = tmp.name
        with st.spinner("🔍 Analyzing..."):
            prob = predict(tmp_path)
        os.unlink(tmp_path)
        show_result(prob)
