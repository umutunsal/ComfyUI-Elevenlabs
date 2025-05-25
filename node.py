import requests
import json
import time
import io
import torch
import torchaudio

class ElevenLabsNode:
    voices_cache = None
    models_cache = None
    last_fetch_time = 0
    cache_duration = 3600  # Cache duration in seconds (1 hour)

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        voices = cls.fetch_elevenlabs_voices()
        models = cls.fetch_elevenlabs_models()
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello, how are you?"
                }),
                "voice": (voices,),
                "model": (models,),
                "stability": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "similarity_boost": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "style": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "use_speaker_boost": ("BOOLEAN", {
                    "default": True,
                }),
            },
            "optional": {
                "input_text": ("STRING", {"forceInput": True}),
                
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate_speech"
    CATEGORY = "ElevenLabs"

    @classmethod
    def fetch_elevenlabs_voices(cls):
        current_time = time.time()
        if cls.voices_cache is None or (current_time - cls.last_fetch_time > cls.cache_duration):
            voice_list = []

            # Fetch voices from the API
            url = "https://api.elevenlabs.io/v1/voices"
            try:
                response = requests.get(url)
                response.raise_for_status()
                voices = response.json()["voices"]
                voice_list = [f"{voice['name']} ({voice['voice_id']})" for voice in voices]
                cls.voices_cache = voice_list
                cls.last_fetch_time = current_time
            except requests.exceptions.RequestException as e:
                print(f"Error fetching voices: {e}")
                if cls.voices_cache is None:
                    cls.voices_cache = ["error_fetching_voices"]
        return cls.voices_cache

    @classmethod
    def fetch_elevenlabs_models(cls):
        current_time = time.time()
        if cls.models_cache is None or (current_time - cls.last_fetch_time > cls.cache_duration):
            cls.models_cache = ["eleven_multilingual_v2", "eleven_english_sts_v2", "eleven_turbo_v2"]
            cls.last_fetch_time = current_time
        return cls.models_cache

    def ensure_3d_tensor(self, tensor):
        if tensor.dim() == 1:
            return tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 2:
            return tensor.unsqueeze(0)
        elif tensor.dim() > 3:
            return tensor.squeeze().unsqueeze(0)
        return tensor

    def generate_speech(self, api_key, text, voice, model, stability, similarity_boost, style, use_speaker_boost, input_text=None, input_audio=None):
        # Use input_text if provided, otherwise use the text from the textbox
        final_text = input_text if input_text is not None else text

        voice_id = voice.split("(")[-1].strip(")")

        headers = {
            "Accept": "application/json",
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }

        if input_audio is not None:
            # Speech-to-Speech
            url = f"https://api.elevenlabs.io/v1/speech-to-speech/{voice_id}"
            payload = {
                "model_id": model,
                "voice_settings": {
                    "stability": stability,
                    "similarity_boost": similarity_boost,
                    "style": style,
                    "use_speaker_boost": use_speaker_boost
                }
            }
            
            input_waveform = self.ensure_3d_tensor(input_audio["waveform"])
            
            wav_buffer = io.BytesIO()
            torchaudio.save(wav_buffer, input_waveform.squeeze(0), input_audio["sample_rate"], format="wav")
            wav_buffer.seek(0)
            
            files = {"audio": ("input.wav", wav_buffer, "audio/wav")}
            
            try:
                response = requests.post(url, headers=headers, data=payload, files=files)
            except requests.exceptions.RequestException as e:
                print(f"Error in speech-to-speech: {str(e)}")
                return ({"waveform": torch.zeros(1, 1, 1).float(), "sample_rate": input_audio["sample_rate"]},)
        else:
            # Text-to-Speech
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            payload = {
                "text": final_text,
                "model_id": model,
                "voice_settings": {
                    "stability": stability,
                    "similarity_boost": similarity_boost,
                    "style": style,
                    "use_speaker_boost": use_speaker_boost
                }
            }
            try:
                response = requests.post(url, headers=headers, json=payload)
            except requests.exceptions.RequestException as e:
                print(f"Error in text-to-speech: {str(e)}")
                return ({"waveform": torch.zeros(1, 1, 1).float(), "sample_rate": 44100},)

        if response.status_code == 200:
            audio_content = io.BytesIO(response.content)
            
            try:
                waveform, sample_rate = torchaudio.load(audio_content)
            except Exception as e:
                print(f"Error loading audio content: {str(e)}")
                return ({"waveform": torch.zeros(1, 1, 1).float(), "sample_rate": 44100},)
            
            waveform = self.ensure_3d_tensor(waveform)
            
            if waveform.dtype != torch.float32:
                waveform = waveform.float() / torch.iinfo(waveform.dtype).max
            
            # Ensure the waveform is 3D
            if waveform.dim() != 3:
                print(f"Warning: Unexpected tensor dimension {waveform.dim()}. Reshaping to 3D.")
                waveform = waveform.view(1, 1, -1)
            
            return ({"waveform": waveform, "sample_rate": sample_rate},)
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return ({"waveform": torch.zeros(1, 1, 1).float(), "sample_rate": 44100},)

    @classmethod
    def IS_CHANGED(cls, api_key, text, voice, model, stability, similarity_boost, style, use_speaker_boost, input_text=None, input_audio=None):
        return (api_key, text, voice, model, stability, similarity_boost, style, use_speaker_boost, input_text, input_audio)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "ElevenLabsNode": ElevenLabsNode
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "ElevenLabsNode": "ElevenLabs TTS Node"
}
