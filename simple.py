import os
import io
import re
from pathlib import Path
import textwrap
import concurrent.futures
from tempfile import NamedTemporaryFile

import gradio as gr
from openai import OpenAI
from pypdf import PdfReader
from loguru import logger

# Maximum text length for OpenAI TTS API (4096 characters, but using 4000 to be safe)
MAX_TTS_LENGTH = 4000

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
    text = re.sub(r'\n\s*[A-Za-z0-9_\-\.]+\s*\|\s*[A-Za-z0-9_\-\.]+\s*\n', '\n', text)
    
    # Fix multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

def split_text_into_chunks(text, max_length=MAX_TTS_LENGTH):
    """
    Split text into chunks that are small enough for the TTS API.
    Try to split at paragraph or sentence boundaries where possible.
    """
    # If text is already short enough, return it as a single chunk
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    
    # First try to split by paragraphs (double newlines)
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    for paragraph in paragraphs:
        # If adding this paragraph would exceed the limit, save current chunk and start a new one
        if len(current_chunk) + len(paragraph) + 2 > max_length:
            # If current paragraph is too long by itself, split by sentences
            if len(paragraph) > max_length:
                # Try to split by sentences (period followed by space)
                sentences = re.split(r'(?<=\. )', paragraph)
                
                # Add sentences to current chunk until it's full
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 > max_length:
                        # If current sentence is still too long, split by words
                        if len(sentence) > max_length:
                            words = sentence.split(' ')
                            for word in words:
                                if len(current_chunk) + len(word) + 1 > max_length:
                                    chunks.append(current_chunk.strip())
                                    current_chunk = word + " "
                                else:
                                    current_chunk += word + " "
                        else:
                            # Add chunk and start new one with this sentence
                            chunks.append(current_chunk.strip())
                            current_chunk = sentence + " "
                    else:
                        current_chunk += sentence + " "
            else:
                # Add chunk and start new one with this paragraph
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        else:
            current_chunk += paragraph + "\n\n"
    
    # Add the final chunk if there's anything left
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def generate_audio_chunk(text, api_key, voice="alloy", model="tts-1"):
    """Generate audio for a single chunk of text"""
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text
        )
        
        return response.content
    except Exception as e:
        logger.error(f"Error generating audio for chunk: {e}")
        raise

def process_pdf(pdf_file, api_key, voice, tts_model):
    """Process the PDF and generate audio"""
    if not api_key:
        return None, "OpenAI API key is required", None
    
    try:
        # Extract text from PDF
        logger.info("Extracting text from PDF")
        text = extract_text_from_pdf(pdf_file.name)
        
        # Clean the text (minimal processing)
        logger.info("Cleaning text")
        cleaned_text = clean_text(text)
        
        # Split text into chunks
        logger.info("Splitting text into chunks")
        chunks = split_text_into_chunks(cleaned_text)
        logger.info(f"Text split into {len(chunks)} chunks")
        
        # Generate audio for each chunk in parallel
        logger.info(f"Generating audio with voice '{voice}' using model '{tts_model}'")
        
        # Create a temporary directory for audio chunks
        temp_dir = "./temp_audio_chunks"
        os.makedirs(temp_dir, exist_ok=True)
        
        audio_data = b""
        
        # Process chunks in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all tasks
            futures = [
                executor.submit(generate_audio_chunk, chunk, api_key, voice, tts_model)
                for chunk in chunks
            ]
            
            # Collect results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    chunk_audio = future.result()
                    audio_data += chunk_audio
                    logger.info(f"Processed chunk {i+1}/{len(chunks)}")
                except Exception as e:
                    logger.error(f"Error processing chunk {i+1}: {e}")
                    raise
        
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