import os
import io
import re
from pathlib import Path

import gradio as gr
from openai import OpenAI
from pypdf import PdfReader
from loguru import logger

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file with minimal processing"""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise

def clean_text(text):
    """
    Perform minimal cleaning on the text to improve readability
    without changing the actual content:
    - Remove page numbers
    - Fix common formatting issues
    """
    # Remove common page number patterns
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Remove standalone page numbers (e.g., "42" on its own line)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Remove header/footer patterns if they appear on every page
    # This is a simple version - might need to be customized based on specific PDFs
    text = re.sub(r'\n\s*[A-Za-z0-9_\-\.]+\s*\|\s*[A-Za-z0-9_\-\.]+\s*\n', '\n', text)
    
    # Fix multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

def generate_audio(text, api_key, voice="alloy", model="tts-1"):
    """Generate audio directly from text using OpenAI TTS"""
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text
        )
        
        return response.content
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        raise

def process_pdf(pdf_file, api_key, voice, tts_model):
    """Process the PDF and generate audio"""
    if not api_key:
        return None, "OpenAI API key is required", None
    
    try:
        # Extract text from PDF
        logger.info("Extracting text from PDF")
        text = extract_text_from_pdf(pdf_file.name)
        
        # Clean the text
        logger.info("Cleaning text")
        cleaned_text = clean_text(text)
        
        # Generate audio
        logger.info(f"Generating audio with voice '{voice}' using model '{tts_model}'")
        audio_data = generate_audio(cleaned_text, api_key, voice, tts_model)
        
        # Create a temporary file for the audio
        temp_audio_path = "temp_audio.mp3"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_data)
        
        return temp_audio_path, None, cleaned_text
    
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        logger.error(error_msg)
        return None, error_msg, None

def create_ui():
    """Create and launch the Gradio UI"""
    with gr.Blocks(title="PDF to Audio - Direct TTS") as demo:
        gr.Markdown("# PDF to Audio Converter")
        gr.Markdown("Upload a PDF file and convert it directly to audio. The text will be read word-for-word with minimal cleaning.")
        
        with gr.Row():
            with gr.Column():
                pdf_file = gr.File(label="Upload PDF")
                api_key = gr.Textbox(
                    label="OpenAI API Key", 
                    placeholder="Enter your OpenAI API key", 
                    type="password"
                )
                voice = gr.Dropdown(
                    label="Voice",
                    choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                    value="alloy"
                )
                tts_model = gr.Dropdown(
                    label="TTS Model",
                    choices=["tts-1", "tts-1-hd"],
                    value="tts-1"
                )
                submit_btn = gr.Button("Generate Audio")
            
            with gr.Column():
                audio_output = gr.Audio(label="Audio Output", type="filepath")
                error_output = gr.Textbox(label="Error", visible=False)
                text_output = gr.Textbox(label="Extracted Text", lines=10)
        
        submit_btn.click(
            fn=process_pdf,
            inputs=[pdf_file, api_key, voice, tts_model],
            outputs=[audio_output, error_output, text_output]
        ).then(
            fn=lambda error: gr.update(visible=bool(error)) if error else gr.update(visible=False),
            inputs=[error_output],
            outputs=[error_output]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch()