# English Accent Detector

This tool classifies the speaker's English accent (e.g., British, American, Australian) from any public video URL (including YouTube and direct `.mp4` links). It uses a pre-trained model from Hugging Face to identify the accent and outputs a confidence score along with an explanation.

## Features

- Accepts YouTube or public video URLs
- Automatically downloads the video and extracts audio
- Classifies the accent using a pretrained model
- Outputs:
  - Predicted accents  
    •	American  
    •	Australian  
    •	Canadian  
    •	Chinese  
    •	Czech  
    •	Dutch  
    •	Eastern european  
    •	English  
    •	Estonian  
    •	Finnish  
    •	French  
    •	German  
    •	Hungarian  
    •	Indian  
    •	Irish  
    •	Italian  
    •	Jamaican  
    •	Latin american  
    •	Malaysian  
    •	New zealand  
    •	Polish  
    •	Romanian  
    •	"Scottish  
    •	Singaporean  
    •	Slovak  
    •	South african  
    •	Spanish  
    •	Welsh   
  - Confidence score (0–100%)
  - Explanation of the prediction

## Pretrained Model

This tool uses the Hugging Face model:
`ylacombe/accent-classifier`  
https://huggingface.co/ylacombe/accent-classifier

## Installation

1. Clone the repository:  
   Git Clone : git@github.com:imant686/english-accent-detector.git
2. Install dependencies:  
   `pip install -r requirements.txt`      
3. Install ffmpeg (required for audio extraction):  
#### 	•	**macOS**   
     `brew install ffmpeg`  
#### •	**Ubuntu**   
   `sudo apt install ffmpeg`
#### •  **Windows**  
   Download from https://ffmpeg.org/download.html and add the bin folder to your system PATH.

## Running the App  
Run the Streamlit app with:  
`streamlit run app.py`  

## Example Usage    
	1.	Paste any public YouTube or direct video URL into the input box.  
	2.	Click “Analyse”.  
	3.	View the predicted accent and confidence score. 

## How It Works
  • Video download is handled using yt_dlp.    
  • Audio extraction is performed using pydub with ffmpeg.  
  • The transformers model classifies the accent based on audio input.  
  • The tool returns the top predicted accent and its confidence score.  

## Dependencies 
`streamlit`    
`torch`    
`torchaudio`    
`transformers`    
`pydub`    
`requests`    
`ffmpeg-python`    
`scikit-learn`    
`numpy`

  
