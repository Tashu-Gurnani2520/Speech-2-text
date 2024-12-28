import gradio as gr
from transformers import pipeline

# Load the Whisper model for Hindi transcription
pipe = pipeline(model="Tashuu/whisper-medium-hindi")

def transcribe_audio(audio):
    """
    Transcribes the uploaded audio file using the fine-tuned Whisper model.
    """
    try:
        transcription = pipe(audio)["text"]
        return transcription
    except Exception as e:
        return f"Error during transcription: {str(e)}"

# Create the Gradio interface as a serverless function
def handler(event, context):
    interface = gr.Interface(
        fn=transcribe_audio,
        inputs=gr.Audio(type="filepath"),
        outputs="text",
        live=False
    )
    return interface.launch(share=True, inline=True)
