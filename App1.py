import streamlit as st
import whisper
import tempfile
import os
import json
import ffmpeg

def format_time(seconds):
    """ Convert seconds to SRT time format (HH:MM:SS,ms) """
    millis = int((seconds - int(seconds)) * 1000)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

def transcribe_audio(file_path):
    """ Transcribes an audio/video file and saves output in TXT, SRT, and JSON formats. """
    
    # Load Whisper model
    model = whisper.load_model("small")

    # Transcribe the audio/video
    result = model.transcribe(file_path, language="en")  # Force transcription in English

    # Get file name without extension
    base_name = os.path.splitext(file_path)[0]

    # Save TXT file
    txt_file = base_name + "_transcript.txt"
    with open(txt_file, "w", encoding="utf-8") as txt:
        txt.write(result["text"])

    # Save SRT file
    srt_file = base_name + "_subtitles.srt"
    with open(srt_file, "w", encoding="utf-8") as srt:
        for i, segment in enumerate(result["segments"]):
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"]
            srt.write(f"{i+1}\n{format_time(start_time)} --> {format_time(end_time)}\n{text}\n\n")

    # Save JSON file
    json_file = base_name + "_transcript.json"
    with open(json_file, "w", encoding="utf-8") as json_out:
        json.dump(result, json_out, indent=4, ensure_ascii=False)

    return txt_file, srt_file, json_file, result["text"]

# Streamlit UI
st.title("🎙 Whisper AI Transcription App")

# Upload Video or Audio File
uploaded_file = st.file_uploader("Upload an audio or video file", 
                                 type=["mp3", "wav", "m4a", "mp4", "mov", "avi", "flv", "mkv"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_filename = temp_file.name

    # Check if the file is a video
    file_extension = os.path.splitext(temp_filename)[1].lower()
    
    if file_extension in [".mp4", ".mov", ".avi", ".flv", ".mkv"]:
        st.video(uploaded_file)  # Show the uploaded video
        
        # Convert video to audio
        audio_filename = temp_filename + ".wav"
        try:
            ffmpeg.input(temp_filename).output(audio_filename, format='wav').run(overwrite_output=True)
        except ffmpeg.Error as e:
            st.error(f"FFmpeg error: {e.stderr.decode('utf-8')}")
            os.remove(temp_filename)
            st.stop()
    else:
        st.audio(uploaded_file, format="audio/wav")  # Show the uploaded audio
        audio_filename = temp_filename  # Use directly if it's already audio

    # Transcribe the extracted audio
    st.write("🔄 Transcribing... Please wait ⏳")
    
    try:
        txt_file, srt_file, json_file, transcription_text = transcribe_audio(audio_filename)
    except Exception as e:
        st.error(f"Transcription error: {e}")
        os.remove(temp_filename)
        if temp_filename != audio_filename:
            os.remove(audio_filename)
        st.stop()

    # Display Transcription
    st.subheader("Transcription:")
    st.write(transcription_text)

    # Provide Download Links
    st.subheader("📂 Download Transcription Files:")
    
    with open(txt_file, "rb") as f:
        st.download_button(label="📥 Download TXT", data=f, file_name="transcription.txt", mime="text/plain")

    with open(srt_file, "rb") as f:
        st.download_button(label="📥 Download SRT", data=f, file_name="subtitles.srt", mime="text/plain")

    with open(json_file, "rb") as f:
        st.download_button(label="📥 Download JSON", data=f, file_name="transcription.json", mime="application/json")

    # Delete temp files
    os.remove(temp_filename)
    if temp_filename != audio_filename:
        os.remove(audio_filename)
