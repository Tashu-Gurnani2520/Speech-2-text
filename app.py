from transformers import pipeline
import gradio as gr

# Load the fine-tuned Whisper model
pipe = pipeline(model="Tashuu/whisper-medium-hindi")

def transcribe_audio(audio):
    """
    Transcribes the uploaded audio file using the fine-tuned Whisper model.
    """
    try:
        # Use the pipeline to process the audio file
        transcription = pipe(audio)["text"]
        return transcription
    except Exception as e:
        return f"Error during transcription: {str(e)}"

# Define Gradio interface
interface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath"),  # Upload audio file
    outputs="text",  # Display transcription text
    title="Hindi Speech-to-Text Transcription",
    description=(
        "Upload an audio file in Hindi, and this app will transcribe it to text using "
        "a fine-tuned Whisper model. Supports diverse Hindi accents and pronunciations."
    ),
    theme="default",
    live=False  # Set True for real-time transcription (may require tweaking)
)

# Launch the web app
if __name__ == "__main__":
    interface.launch(share=True)  # share=True to generate a public link
