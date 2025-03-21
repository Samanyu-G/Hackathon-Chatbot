import os
import sys
import random
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import importlib.util

# Function to check if module is installed
def is_module_installed(module_name):
    return importlib.util.find_spec(module_name) is not None

# Install required dependencies if missing
required_modules = {
    "speech_recognition": "SpeechRecognition",
    "pyttsx3": "pyttsx3",
    "deep_translator": "deep-translator",
    "torch": "torch",
    "transformers": "transformers",
    "sv_ttk": "sv-ttk",
    "PIL": "pillow",
    "ttkbootstrap": "ttkbootstrap",
    "langdetect": "langdetect"  # Added for better language detection
}

missing_modules = []
for module, pip_name in required_modules.items():
    if not is_module_installed(module):
        missing_modules.append(pip_name)

if missing_modules:
    print(f"Installing missing modules: {', '.join(missing_modules)}")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_modules)
        print("Please restart the application")
        sys.exit(0)
    except subprocess.CalledProcessError:
        print("Failed to install required dependencies. Please install them manually.")
        sys.exit(1)

# Now import the modules
import speech_recognition as sr
import pyttsx3
from deep_translator import GoogleTranslator
from langdetect import detect
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import sv_ttk
from PIL import Image, ImageTk

# Configuration class
class Config:
    # App settings
    APP_NAME = "AI Voice Assistant"
    VERSION = "1.1.0"
    
    # Model settings
    DEFAULT_DIALOGUE_MODEL = "microsoft/DialoGPT-medium"
    TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-{}-{}"  # Format: src-tgt
    
    # GUI settings
    WINDOW_SIZE = "900x700"
    PADDING = 10
    CHAT_FONT = ("Segoe UI", 11)
    
    # TTS settings
    DEFAULT_RATE = 175
    DEFAULT_VOLUME = 1.0
    
    # Icons (Emoji representations)
    ICONS = {
        "chef": "👨‍🍳",
        "teacher": "👩‍🏫",
        "actor": "🎭",
        "banker": "💼",
        "mic_on": "🎤",
        "mic_off": "⏹️",
        "send": "📤",
        "light": "☀️",
        "dark": "🌙"
    }

# Logger for better debugging
class Logger:
    def __init__(self, log_file="assistant_log.txt"):
        self.log_file = log_file
        self._ensure_log_dir()
        
    def _ensure_log_dir(self):
        os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else '.', exist_ok=True)
    
    def log(self, message, level="INFO"):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        # Print to console
        print(log_entry, end="")
        
        # Write to file
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Error writing to log file: {e}")

# Improved PersonalityBot class
class PersonalityBot:
    def __init__(self, name, background, speaking_style, knowledge_domains, catchphrases, avatar=None, voice_settings=None):
        self.name = name
        self.background = background
        self.speaking_style = speaking_style
        self.knowledge_domains = knowledge_domains
        self.catchphrases = catchphrases
        self.avatar = avatar or self._get_default_avatar()
        self.voice_settings = voice_settings or {}
        self.logger = Logger()
        
        # Initialize models lazily to save memory
        self._tokenizer = None
        self._model = None
        
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            try:
                self.logger.log(f"Loading tokenizer for {self.name}")
                self._tokenizer = AutoTokenizer.from_pretrained(Config.DEFAULT_DIALOGUE_MODEL)
            except Exception as e:
                self.logger.log(f"Error loading tokenizer: {e}", "ERROR")
                raise
        return self._tokenizer
    
    @property
    def model(self):
        if self._model is None:
            try:
                self.logger.log(f"Loading model for {self.name}")
                self._model = AutoModelForCausalLM.from_pretrained(Config.DEFAULT_DIALOGUE_MODEL)
            except Exception as e:
                self.logger.log(f"Error loading model: {e}", "ERROR")
                raise
        return self._model
    
    def _get_default_avatar(self):
        if "Chef" in self.name:
            return Config.ICONS["chef"]
        elif "Professor" in self.name or "Teacher" in self.name:
            return Config.ICONS["teacher"]
        elif "Actor" in self.name:
            return Config.ICONS["actor"]
        elif "Banker" in self.name:
            return Config.ICONS["banker"]
        else:
            return "🤖"
    
    def is_query_relevant(self, query):
        """Check if the query is relevant to this personality's knowledge domains."""
        query_lower = query.lower()
        return any(domain.lower() in query_lower for domain in self.knowledge_domains)
    
    def process_query(self, query, language="en"):
        """Process a user query according to the bot's personality."""
        self.logger.log(f"{self.name} processing query: {query}")
        
        # Check relevance
        relevant = self.is_query_relevant(query)
        
        if not relevant:
            return self.get_off_topic_response(language)
        
        try:
            # Generate a response
            response = self.generate_response(query)
            
            # Add a catchphrase sometimes
            if random.random() < 0.3:  # 30% chance
                response += " " + random.choice(self.catchphrases)
            
            return response
        except Exception as e:
            self.logger.log(f"Error processing query: {e}", "ERROR")
            return f"I'm having trouble processing that request. {random.choice(self.catchphrases)}"
    
    def get_off_topic_response(self, language):
        """Return a response when the query is off-topic for this personality."""
        responses = {
            "chef": [
                "I'd rather talk about food! Ask me about recipes or cooking techniques.",
                "That's not on my menu. I'm here to discuss culinary matters!",
                "I'm a chef, not a general knowledge bot. Let's talk about cuisine!"
            ],
            "teacher": [
                "I'd be happy to help with academic questions, but that's outside my expertise.",
                "Let's focus on educational topics. I'm here to help you learn!",
                "As an educator, I prefer to discuss topics related to learning and knowledge."
            ],
            "actor": [
                "Darling, I'm a Bollywood star! I'd prefer to talk about films and drama!",
                "Oh my! Such a boring question! Ask me about movies, dancing, or my FABULOUS career!",
                "WHAT? That's not about Bollywood! Let's talk about something more... DRAMATIC!"
            ],
            "banker": [
                "Look, I don't have time for this. Can we stick to financial matters?",
                "Unless it's about money, I'm not interested. Time is money, you know.",
                "Do I look like I care about that? Ask me about something financial."
            ]
        }
        
        personality_type = (
            "chef" if "Chef" in self.name 
            else "teacher" if "Professor" in self.name or "Teacher" in self.name
            else "actor" if "Actor" in self.name 
            else "banker"
        )
        
        return random.choice(responses[personality_type])
    
    def generate_response(self, query):
        """Generate a response using the language model with the bot's personality injected."""
        try:
            # Create a personalized prompt
            prompt = f"As {self.name}, {self.background}, speaking in a {self.speaking_style} manner: {query}"
            
            # Tokenize the prompt
            inputs = self.tokenizer.encode(prompt + self.tokenizer.eos_token, return_tensors='pt') # type: ignore
            
            # Generate a response
            with torch.no_grad():  # Disable gradient calculation for inference
                reply_ids = self.model.generate(
                    inputs, 
                    max_length=250, 
                    pad_token_id=self.tokenizer.eos_token_id,
                    temperature=0.8,  # Add some randomness
                    top_p=0.92,  # Use nucleus sampling
                    no_repeat_ngram_size=2  # Avoid repetition
                )
            
            # Decode the response
            response = self.tokenizer.decode(reply_ids[0], skip_special_tokens=True)
            
            # Extract only the assistant's reply (remove the prompt)
            if query in response:
                response = response.split(query, 1)[1].strip()
            
            self.logger.log(f"Generated response: {response}")
            return response
        except Exception as e:
            self.logger.log(f"Error generating response: {e}", "ERROR")
            return "I'm having trouble formulating a response right now."

# Improved VoiceAssistant class
class VoiceAssistant:
    def __init__(self):
        self.logger = Logger()
        self.recognizer = sr.Recognizer()
        self.engine = self.initialize_tts_engine()
        self.is_listening = False
        self.current_language = "en"
        self.conversation_history = []

        self.logger.log("Initializing personalities")
        self.personalities = self.create_personalities()
        self.active_personality = self.personalities[0]  # Default personality

        self.list_voices()  # Call the function here to see available voices on startup
        
        # Set up language detection
        try:
            from langdetect import DetectorFactory
            DetectorFactory.seed = 0  # For consistent language detection
        except ImportError:
            self.logger.log("Langdetect not available, falling back to basic detection", "WARNING")

    def create_personalities(self):
        """Create the personality bots."""
        return [
            PersonalityBot(
                name="Chef Antonio",
                background="A passionate Italian chef with 30 years of culinary experience",
                speaking_style="Enthusiastic and food-obsessed",
                knowledge_domains=["food", "cooking", "recipes", "ingredients", "cuisine", "kitchen", "restaurants", "meal", "dinner", "lunch", "breakfast"],
                catchphrases=["Mamma mia!", "That's a-spicy!", "The secret is always in the sauce!"],
                voice_settings={"rate": 220, "volume": 2.0}
            ),
            PersonalityBot(
                name="Professor Amelia",
                background="A knowledgeable university professor with multiple PhDs",
                speaking_style="Articulate and informative",
                knowledge_domains=["science", "history", "literature", "mathematics", "education", "research", "study", "learn", "book", "university", "academic"],
                catchphrases=["Fascinating question!", "Let me explain that concept.", "Knowledge is power!"],
                voice_settings={"rate": 160, "volume": 0.9}
            ),
            PersonalityBot(
                name="Bollywood Actor Raj",
                background="A dramatic and popular Bollywood star",
                speaking_style="Over-dramatic and exuberant",
                knowledge_domains=["movies", "acting", "dance", "drama", "entertainment", "bollywood", "film", "cinema", "performance", "actor", "actress"],
                catchphrases=["Oh my goodness!", "So DRAMATIC!", "This reminds me of my blockbuster film!"],
                voice_settings={"rate": 180, "volume": 1.6, "voice_id": 1}
            ),
            PersonalityBot(
                name="Banker Morgan",
                background="A rude investment banker who is always busy",
                speaking_style="Impatient and condescending",
                knowledge_domains=["money", "finance", "investment", "banking", "stocks", "wealth", "economy", "business", "market", "profit", "loss"],
                catchphrases=["Time is money!", "Do I look like I care?", "Just check the numbers yourself!"],
                voice_settings={"rate": 175, "volume": 1.0}
            )
        ]

    def initialize_tts_engine(self):
        """Initialize the text-to-speech engine."""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', Config.DEFAULT_RATE)
            engine.setProperty('volume', Config.DEFAULT_VOLUME)
            return engine
        except Exception as e:
            self.logger.log(f"Failed to initialize TTS engine: {e}", "ERROR")
            return None
    
    def list_voices(self):
        """List available voices in the system."""
        voices = self.engine.getProperty('voices') # type: ignore
        for index, voice in enumerate(voices):
            print(f"Voice {index}: {voice.name} - ID: {voice.id}")

    def listen(self):
        """Listen to the user and return the recognized text."""
        try:
            with sr.Microphone() as source:
                self.logger.log("Listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5) # type: ignore
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
            try:
                text = self.recognizer.recognize_google(audio) # type: ignore
                self.logger.log(f"Recognized: {text}")
                return text
            except sr.UnknownValueError:
                self.logger.log("Could not understand audio")
                return None
            except sr.RequestError as e:
                self.logger.log(f"Error with the speech recognition service: {e}", "ERROR")
                return None
        except Exception as e:
            self.logger.log(f"Error during listening: {e}", "ERROR")
            return None
    
    def detect_language(self, text):
        """Detect the language of the input text."""
        if not text:
            return "en"  # Default to English
        
        try:
            # Use langdetect for better language detection
            detected = detect(text)
            self.logger.log(f"Detected language: {detected}")
            return detected
        except Exception as e:
            self.logger.log(f"Language detection error: {e}", "WARNING")
            
            # Fallback to simple heuristic
            try:
                translator = GoogleTranslator(source='auto', target='en')
                translated = translator.translate(text)
                
                if translated.lower() != text.lower() and len(translated) > 5:
                    return "non-en"  # Non-English detected
                return "en"
            except Exception as e:
                self.logger.log(f"Translation error: {e}", "ERROR")
                return "en"  # Default to English if detection fails
    
    def translate_text(self, text, source_language="en", target_language="en"):
        """Translate text between languages."""
        if not text or source_language == target_language:
            return text
        
        try:
            translator = GoogleTranslator(source=source_language, target=target_language)
            translated = translator.translate(text)
            self.logger.log(f"Translated from {source_language} to {target_language}")
            return translated
        except Exception as e:
            self.logger.log(f"Translation error: {e}", "ERROR")
            return text
    
    def speak(self, text, language="en"):
        """Convert text to speech in the specified language."""
        if not text or not self.engine:
            return
        
        try:
            self.engine.stop()

            # Set voice properties based on language and personality
            voices = self.engine.getProperty('voices')
            
            # Try to find an appropriate voice
            voice_id = None
            if language.startswith('es'):
                spanish_voices = [v for v in voices if 'spanish' in v.name.lower()]
                if spanish_voices:
                    voice_id = spanish_voices[0].id
            elif language.startswith('fr'):
                french_voices = [v for v in voices if 'french' in v.name.lower()]
                if french_voices:
                    voice_id = french_voices[0].id
            elif language.startswith('de'):
                german_voices = [v for v in voices if 'german' in v.name.lower()]
                if german_voices:
                    voice_id = german_voices[0].id
                    
            # Apply personality-specific voice settings
            if hasattr(self.active_personality, 'voice_settings'):
                if 'rate' in self.active_personality.voice_settings:
                    self.engine.setProperty('rate', self.active_personality.voice_settings['rate'])
                if 'volume' in self.active_personality.voice_settings:
                    self.engine.setProperty('volume', self.active_personality.voice_settings['volume'])
            
            # Set the voice if one was found
            if voice_id:
                self.engine.setProperty('voice', voice_id)
            elif voices:
                # Default to a voice based on the personality (alternating between male/female)
                personality_index = self.personalities.index(self.active_personality)
                voice_index = personality_index % len(voices)
                self.engine.setProperty('voice', voices[voice_index].id)

            # Apply a male voice for Raj
            if self.active_personality.name == "Bollywood Actor Raj":
                male_voice_id = voices[1].id  # Adjust based on `list_voices()`
                self.engine.setProperty('voice', male_voice_id)

            # Apply speed & volume settings
            self.engine.setProperty('rate', self.active_personality.voice_settings.get('rate', 180))
            self.engine.setProperty('volume', self.active_personality.voice_settings.get('volume', 1.0))

            self.engine.say(text)
            self.engine.runAndWait()
            
            
        except Exception as e:
            self.logger.log(f"TTS error: {e}", "ERROR")
    
    def process_query(self, query):
        """Process the query with the active personality."""
        if not query:
            return "I couldn't hear you clearly. Could you please repeat?"
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Detect language
        detected_language = "en"  # Force English

        self.current_language = detected_language
        
        # Translate to English if needed
        if detected_language != "en":
            english_query = self.translate_text(query, detected_language, "en")
        else:
            english_query = query
        
        # Get response from active personality
        english_response = self.active_personality.process_query(english_query)
        
        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": english_response})
        
        # Translate response back to original language if needed
        if detected_language != "en":
            response = self.translate_text(english_response, "en", detected_language)
        else:
            response = english_response
        
        return response
    
    def set_personality(self, index):
        """Set the active personality by index."""
        if 0 <= index < len(self.personalities):
            self.active_personality = self.personalities[index]
            self.logger.log(f"Switched to {self.active_personality.name}")
            return f"Switched to {self.active_personality.name}"
        return "Invalid personality selection"
    
    def start_listening(self):
        """Start the continuous listening loop."""
        self.is_listening = True
        self.logger.log("Started continuous listening")
    
    def stop_listening(self):
        """Stop the listening loop."""
        self.is_listening = False
        self.logger.log("Stopped continuous listening")

# Improved ModernAssistantGUI class
class ModernAssistantGUI:
    def __init__(self, root, assistant):
        self.root = root
        self.root.title(Config.APP_NAME)
        self.root.geometry(Config.WINDOW_SIZE)
        self.root.minsize(600, 400)  # Set minimum window size
        
        self.assistant = assistant
        self.listening_thread = None
        self.theme = "dark"
        self.logger = Logger()
        
        # Load theme
        sv_ttk.set_theme("dark")
        
        # Create UI elements
        self.create_widgets()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Load custom styles
        self.load_styles()
    
    def load_styles(self):
        """Load custom styles for the application."""
        style = ttk.Style()
        
        # Define custom styles
        style.configure("TButton", padding=6)
        style.configure("Accent.TButton", background="#007bff")
        style.configure("Header.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("Subheader.TLabel", font=("Segoe UI", 12))
    
    def create_widgets(self):
        # Main frame with padding
        self.main_frame = ttk.Frame(self.root, padding=Config.PADDING)
        self.main_frame.pack(fill="both", expand=True)
        
        # Create header with logo and title
        self.create_header(self.main_frame)
        
        # Create a notebook for different sections
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill="both", expand=True, pady=10)
        
        # Create the main chat tab
        chat_tab = ttk.Frame(self.notebook)
        self.notebook.add(chat_tab, text="Chat")
        self.create_chat_tab(chat_tab)
        
        # Create the personalities tab
        personalities_tab = ttk.Frame(self.notebook)
        self.notebook.add(personalities_tab, text="Personalities")
        self.create_personalities_tab(personalities_tab)
        
        # Create settings tab
        settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(settings_tab, text="Settings")
        self.create_settings_tab(settings_tab)
        
        # Create about tab
        about_tab = ttk.Frame(self.notebook)
        self.notebook.add(about_tab, text="About")
        self.create_about_tab(about_tab)
        
        # Status bar at the bottom
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.create_status_bar()
    
    def create_header(self, parent):
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill="x", pady=(0, 10))
        
        # Create a placeholder logo
        logo_frame = ttk.Frame(header_frame, width=60, height=60)
        logo_frame.pack(side="left", padx=10)
        
        logo_label = ttk.Label(logo_frame, text="🎙️", font=("Arial", 30))
        logo_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Title and subtitle
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(side="left", padx=10)
        
        ttk.Label(
            title_frame, 
            text=Config.APP_NAME, 
            style="Header.TLabel"
        ).pack(anchor="w")
        
        ttk.Label(
            title_frame, 
            text="Talk with multiple AI personalities",
            style="Subheader.TLabel"
        ).pack(anchor="w")
    
    def create_chat_tab(self, parent):
        # Split into left and right panes
        paned_window = ttk.PanedWindow(parent, orient="horizontal")
        paned_window.pack(fill="both", expand=True)
        
        # Left pane: Active personality
        left_frame = ttk.Frame(paned_window, padding=10)
        paned_window.add(left_frame, weight=1)
        
        # Personality info card
        self.personality_frame = ttk.LabelFrame(left_frame, text="Active Personality")
        self.personality_frame.pack(fill="x", pady=10, padx=5)
        
        self.avatar_label = ttk.Label(
            self.personality_frame, 
            text=self.assistant.active_personality.avatar, 
            font=("Arial", 40)
        )
        self.avatar_label.pack(pady=10)
        
        self.personality_name = ttk.Label(
            self.personality_frame, 
            text=self.assistant.active_personality.name,
            font=("Arial", 14, "bold")
        )
        self.personality_name.pack(pady=5)
        
        self.personality_background = ttk.Label(
            self.personality_frame, 
            text=self.assistant.active_personality.background,
            wraplength=200
        )
        self.personality_background.pack(pady=5)
        
        self.personality_style = ttk.Label(
            self.personality_frame, 
            text=f"Style: {self.assistant.active_personality.speaking_style}",
            wraplength=200
        )
        self.personality_style.pack(pady=5)
        
        # Knowledge domains
        domains_frame = ttk.LabelFrame(left_frame, text="Knowledge Domains")
        domains_frame.pack(fill="x", pady=10, padx=5)
        
        domains_text = ", ".join(self.assistant.active_personality.knowledge_domains[:5])  # Show first 5 domains
        if len(self.assistant.active_personality.knowledge_domains) > 5:
            domains_text += ", ..."
            
        self.domains_label = ttk.Label(
            domains_frame,
            text=domains_text,
            wraplength=200
        )
        self.domains_label.pack(pady=5)
        
        # Quick selection buttons
        quick_select_frame = ttk.LabelFrame(left_frame, text="Quick Select")
        quick_select_frame.pack(fill="x", pady=10, padx=5)
        
        for i, personality in enumerate(self.assistant.personalities):
            btn = ttk.Button(
                quick_select_frame,
                text=personality.name,
                command=lambda idx=i: self.change_personality(idx)
            )
            btn.pack(fill="x", pady=2, padx=5)
        
        # Right pane: Chat area
        right_frame = ttk.Frame(paned_window, padding=10)
        paned_window.add(right_frame, weight=2)
        
        # Chat display
        chat_display_frame = ttk.LabelFrame(right_frame, text="Conversation")
        chat_display_frame.pack(fill="both", expand=True, pady=5)
        
        self.chat_text = scrolledtext.ScrolledText(
            chat_display_frame,
            wrap="word",
            font=Config.CHAT_FONT,
            bg="#f0f0f0" if self.theme == "light" else "#2d2d2d",
            fg="#000000" if self.theme == "light" else "#ffffff"
        )
        self.chat_text.pack(fill="both", expand=True, pady=5, padx=5)
        self.chat_text.config(state="disabled")
        
        # Input area
        input_frame = ttk.Frame(right_frame)
        input_frame.pack(fill="x", pady=10)
        
        self.mic_button = ttk.Button(
            input_frame,
            text=Config.ICONS["mic_on"],
            width=3,
            command=self.toggle_listening
        )
        self.mic_button.pack(side="left", padx=5)
        
        self.text_entry = ttk.Entry(input_frame)
        self.text_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.text_entry.bind("<Return>", lambda e: self.send_text())
        
        self.send_button = ttk.Button(
            input_frame,
            text="Send",
            command=self.send_text,
            style="Accent.TButton"
        )
        self.send_button.pack(side="left", padx=5)
        
        # Clear chat button
        self.clear_button = ttk.Button(
            input_frame,
            text="Clear",
            command=self.clear_chat
        )
        self.clear_button.pack(side="left", padx=5)
    
    def create_personalities_tab(self, parent):
        self.personalities_notebook = ttk.Notebook(parent)
        self.personalities_notebook.pack(fill="both", expand=True, pady=10, padx=10)
        
        # Create a tab for each personality
        for personality in self.assistant.personalities:
            tab = ttk.Frame(self.personalities_notebook, padding=10)
            self.personalities_notebook.add(tab, text=personality.name)
            
            # Avatar and name header
            header = ttk.Frame(tab)
            header.pack(fill="x", pady=10)
            
            avatar = ttk.Label(header, text=personality.avatar, font=("Arial", 40))
            avatar.pack(side="left", padx=20)
            
            name_frame = ttk.Frame(header)
            name_frame.pack(side="left", padx=10)
            
            ttk.Label(
                name_frame, 
                text=personality.name, 
                font=("Arial", 16, "bold")
            ).pack(anchor="w")
            
            ttk.Label(
                name_frame, 
                text=personality.speaking_style,
                font=("Arial", 12, "italic")
            ).pack(anchor="w")
            
            # Background
            bg_frame = ttk.LabelFrame(tab, text="Background")
            bg_frame.pack(fill="x", pady=10, padx=5)
            
            ttk.Label(
                bg_frame, 
                text=personality.background,
                wraplength=600,
                justify="left",
                padding=10
            ).pack(anchor="w")
            
            # Knowledge domains
            domains_frame = ttk.LabelFrame(tab, text="Knowledge Domains")
            domains_frame.pack(fill="x", pady=10, padx=5)
            
            domains_text = ", ".join(personality.knowledge_domains)
            ttk.Label(
                domains_frame, 
                text=domains_text,
                wraplength=600,
                justify="left",
                padding=10
            ).pack(anchor="w")
            
            # Catchphrases
            catchphrase_frame = ttk.LabelFrame(tab, text="Catchphrases")
            catchphrase_frame.pack(fill="x", pady=10, padx=5)
            
            for phrase in personality.catchphrases:
                ttk.Label(
                    catchphrase_frame, 
                    text=f"• {phrase}",
                    wraplength=600,
                    justify="left",
                    padding=(10, 2)
                ).pack(anchor="w")
            
            # Button to activate this personality
            ttk.Button(
                tab,
                text=f"Activate {personality.name}",
                command=lambda idx=self.assistant.personalities.index(personality): self.change_personality(idx),
                style="Accent.TButton"
            ).pack(pady=10)
    
    def create_settings_tab(self, parent):
        # Create a frame with padding
        settings_frame = ttk.Frame(parent, padding=20)
        settings_frame.pack(fill="both", expand=True)
        
        # TTS Settings
        tts_frame = ttk.LabelFrame(settings_frame, text="Text-to-Speech Settings")
        tts_frame.pack(fill="x", pady=10, padx=5)
        
        # Voice rate slider
        ttk.Label(tts_frame, text="Speech Rate:").pack(anchor="w", pady=5, padx=10)
        
        rate_frame = ttk.Frame(tts_frame)
        rate_frame.pack(fill="x", pady=5, padx=10)
        
        self.rate_var = tk.IntVar(value=Config.DEFAULT_RATE)
        
        ttk.Label(rate_frame, text="Slow").pack(side="left")
        rate_slider = ttk.Scale(
            rate_frame,
            from_=100,
            to=250,
            orient="horizontal",
            variable=self.rate_var,
            command=self.update_tts_settings
        )
        rate_slider.pack(side="left", fill="x", expand=True, padx=10)
        ttk.Label(rate_frame, text="Fast").pack(side="left")
        
        # Voice volume slider
        ttk.Label(tts_frame, text="Volume:").pack(anchor="w", pady=5, padx=10)
        
        volume_frame = ttk.Frame(tts_frame)
        volume_frame.pack(fill="x", pady=5, padx=10)
        
        self.volume_var = tk.DoubleVar(value=Config.DEFAULT_VOLUME)
        
        ttk.Label(volume_frame, text="Low").pack(side="left")
        volume_slider = ttk.Scale(
            volume_frame,
            from_=0.5,
            to=1.0,
            orient="horizontal",
            variable=self.volume_var,
            command=self.update_tts_settings
        )
        volume_slider.pack(side="left", fill="x", expand=True, padx=10)
        ttk.Label(volume_frame, text="High").pack(side="left")
        
        # UI Settings
        ui_frame = ttk.LabelFrame(settings_frame, text="UI Settings")
        ui_frame.pack(fill="x", pady=10, padx=5)
        
        # Theme toggle
        theme_frame = ttk.Frame(ui_frame)
        theme_frame.pack(fill="x", pady=10, padx=10)
        
        ttk.Label(theme_frame, text="Theme:").pack(side="left")
        
        self.theme_var = tk.StringVar(value=self.theme)
        
        ttk.Radiobutton(
            theme_frame,
            text=f"{Config.ICONS['light']} Light",
            variable=self.theme_var,
            value="light",
            command=self.toggle_theme
        ).pack(side="left", padx=10)
        
        ttk.Radiobutton(
            theme_frame,
            text=f"{Config.ICONS['dark']} Dark",
            variable=self.theme_var,
            value="dark",
            command=self.toggle_theme
        ).pack(side="left", padx=10)
        
        # Font size setting
        font_frame = ttk.Frame(ui_frame)
        font_frame.pack(fill="x", pady=10, padx=10)
        
        ttk.Label(font_frame, text="Font Size:").pack(side="left")
        
        self.font_size_var = tk.IntVar(value=11)  # Default font size
        
        font_sizes = [9, 10, 11, 12, 14, 16]
        font_combo = ttk.Combobox(
            font_frame,
            textvariable=self.font_size_var,
            values=font_sizes, # type: ignore
            width=5,
            state="readonly"
        )
        font_combo.pack(side="left", padx=10)
        font_combo.bind("<<ComboboxSelected>>", self.update_font_size)
        
        # Advanced Settings
        adv_frame = ttk.LabelFrame(settings_frame, text="Advanced Settings")
        adv_frame.pack(fill="x", pady=10, padx=5)
        
        # Mic sensitivity
        ttk.Label(adv_frame, text="Microphone Sensitivity:").pack(anchor="w", pady=5, padx=10)
        
        mic_frame = ttk.Frame(adv_frame)
        mic_frame.pack(fill="x", pady=5, padx=10)
        
        self.mic_sensitivity_var = tk.DoubleVar(value=0.5)
        
        ttk.Label(mic_frame, text="Low").pack(side="left")
        mic_slider = ttk.Scale(
            mic_frame,
            from_=0.1,
            to=1.0,
            orient="horizontal",
            variable=self.mic_sensitivity_var
        )
        mic_slider.pack(side="left", fill="x", expand=True, padx=10)
        ttk.Label(mic_frame, text="High").pack(side="left")
        
        # Reset to defaults button
        ttk.Button(
            settings_frame,
            text="Reset to Defaults",
            command=self.reset_settings
        ).pack(pady=20)
    
    def create_about_tab(self, parent):
        about_frame = ttk.Frame(parent, padding=20)
        about_frame.pack(fill="both", expand=True)
        
        # App info
        app_frame = ttk.Frame(about_frame)
        app_frame.pack(fill="x", pady=10)
        
        ttk.Label(
            app_frame, 
            text=Config.APP_NAME,
            font=("Arial", 16, "bold")
        ).pack(anchor="center")
        
        ttk.Label(
            app_frame, 
            text=f"Version {Config.VERSION}",
            font=("Arial", 10)
        ).pack(anchor="center")
        
        # Description
        desc_frame = ttk.LabelFrame(about_frame, text="About This Application")
        desc_frame.pack(fill="x", pady=10, padx=5)
        
        description = (
            "This AI Voice Assistant demonstrates the use of multiple personality bots with "
            "different characters and knowledge domains. The application provides a modern "
            "interface for interacting with these AI personalities through text or voice.\n\n"
            "Features include:\n"
            "• Multiple AI personalities with different expertise\n"
            "• Voice recognition and text-to-speech capabilities\n"
            "• Automatic language detection and translation\n"
            "• Customizable UI settings\n"
            "• Conversation history\n\n"
            "This is a demo application showcasing the integration of various AI and NLP technologies."
        )
        
        ttk.Label(
            desc_frame,
            text=description,
            wraplength=600,
            justify="left",
            padding=10
        ).pack(anchor="w")
        
        # Technologies used
        tech_frame = ttk.LabelFrame(about_frame, text="Technologies Used")
        tech_frame.pack(fill="x", pady=10, padx=5)
        
        technologies = (
            "• Python\n"
            "• Tkinter and ttk for GUI\n"
            "• Transformers for natural language processing\n"
            "• SpeechRecognition for voice input\n"
            "• pyttsx3 for text-to-speech\n"
            "• deep-translator for language translation\n"
            "• Sun Valley ttk theme for modern UI"
        )
        
        ttk.Label(
            tech_frame,
            text=technologies,
            wraplength=600,
            justify="left",
            padding=10
        ).pack(anchor="w")
    
    def create_status_bar(self):
        status_frame = ttk.Frame(self.main_frame)
        status_frame.pack(fill="x", side="bottom")
        
        ttk.Separator(status_frame, orient="horizontal").pack(fill="x", pady=5)
        
        self.status_label = ttk.Label(
            status_frame, 
            textvariable=self.status_var, 
            anchor="w"
        )
        self.status_label.pack(side="left", padx=10)
        
        # Version info on right
        ttk.Label(
            status_frame,
            text=f"v{Config.VERSION}",
            anchor="e"
        ).pack(side="right", padx=10)
    
    def toggle_theme(self):
        self.theme = self.theme_var.get()
        
        if self.theme == "light":
            sv_ttk.set_theme("light")

            # Main background and chat area
            self.root.config(bg="#e3f6ff")  # Light cyan futuristic
            self.chat_text.config(bg="#eef2ff", fg="#0f3460", insertbackground="#0057e7")  # Soft neon blue
            
            # Buttons
            self.mic_button.config(style="Futuristic.TButton")
            self.send_button.config(style="FuturisticAccent.TButton")
            self.clear_button.config(style="FuturisticAccent.TButton")
            
            # Labels
            self.personality_name.config(foreground="#0057e7")  # Electric blue
            self.personality_background.config(foreground="#0f3460")  # Deep sci-fi blue
            self.personality_style.config(foreground="#4a4a4a")  # Dark metallic gray
            
            # Frames & Borders
            self.personality_frame.config(borderwidth=2, relief="ridge", bg="#d6ecff") # type: ignore
            self.domains_label.config(foreground="#1a1a2e")
        
        else:
            sv_ttk.set_theme("dark")

            # Reset to dark futuristic colors
            self.root.config(bg="#1a1a2e")  # Dark cyberpunk blue
            self.chat_text.config(bg="#2d2d2d", fg="#ffffff", insertbackground="#00c8ff")
            
            # Reset other elements
            self.personality_name.config(foreground="#00c8ff")
            self.personality_background.config(foreground="#ffffff")
            self.personality_style.config(foreground="#b0b0b0")
            self.personality_frame.config(borderwidth=1, relief="solid", bg="#202020") # type: ignore

    
    def update_font_size(self, event=None):
        try:
            size = self.font_size_var.get()
            font = ("Segoe UI", size)
            self.chat_text.config(font=font)
        except Exception as e:
            self.logger.log(f"Error updating font size: {e}", "ERROR")
    
    def update_tts_settings(self, event=None):
        try:
            if self.assistant.engine:  # Make sure this line has a colon (:)
                new_rate = self.rate_var.get()
                new_volume = self.volume_var.get()

                self.assistant.engine.setProperty('rate', new_rate)
                self.assistant.engine.setProperty('volume', new_volume)

                self.logger.log(f"Updated TTS settings: Rate={new_rate}, Volume={new_volume}")
                self.status_var.set("TTS settings updated")
        except Exception as e:
            self.logger.log(f"Error updating TTS settings: {e}", "ERROR")


    
    def reset_settings(self):
        # Reset TTS settings
        self.rate_var.set(Config.DEFAULT_RATE)
        self.volume_var.set(Config.DEFAULT_VOLUME)
        self.update_tts_settings()
        
        # Reset UI settings
        self.font_size_var.set(11)
        self.update_font_size()
        
        # Reset theme to dark
        self.theme_var.set("dark")
        self.toggle_theme()
        
        # Reset mic sensitivity
        self.mic_sensitivity_var.set(0.5)
        
        self.status_var.set("Settings reset to defaults")
    
    def change_personality(self, index):
        response = self.assistant.set_personality(index)
        
        # Update personality info
        self.avatar_label.config(text=self.assistant.active_personality.avatar)
        self.personality_name.config(text=self.assistant.active_personality.name)
        self.personality_background.config(text=self.assistant.active_personality.background)
        self.personality_style.config(text=f"Style: {self.assistant.active_personality.speaking_style}")
        
        domains_text = ", ".join(self.assistant.active_personality.knowledge_domains[:5])
        if len(self.assistant.active_personality.knowledge_domains) > 5:
            domains_text += ", ..."
        self.domains_label.config(text=domains_text)
        
        # Add system message to chat
        self.add_message(response, "system")
        
        # Update status
        self.status_var.set(f"Active personality: {self.assistant.active_personality.name}")
    
    def toggle_listening(self):
        if self.assistant.is_listening:
            self.stop_listening()
        else:
            self.start_listening()
    
    def start_listening(self):
        try:
            self.assistant.start_listening()
            self.mic_button.config(text=Config.ICONS["mic_off"])
            self.status_var.set("Listening...")
            
            # Start listening in a separate thread
            if self.listening_thread is None or not self.listening_thread.is_alive():
                self.listening_thread = threading.Thread(target=self.continuous_listen)
                self.listening_thread.daemon = True
                self.listening_thread.start()
        except Exception as e:
            self.logger.log(f"Error starting listening: {e}", "ERROR")
            self.status_var.set("Error: Could not start listening")
    
    def stop_listening(self):
        try:
            self.assistant.stop_listening()
            self.mic_button.config(text=Config.ICONS["mic_on"])
            self.status_var.set("Listening stopped")
        except Exception as e:
            self.logger.log(f"Error stopping listening: {e}", "ERROR")
    
    def continuous_listen(self):
        while self.assistant.is_listening:
            text = self.assistant.listen()
            if text:
                # Schedule processing in the main thread
                self.root.after(0, self.process_voice_input, text)
            time.sleep(0.1)  # Short delay to prevent CPU hogging
    
    def process_voice_input(self, text):
        self.add_message(text, "user")
        self.process_query(text)
    
    def send_text(self):
        text = self.text_entry.get().strip()
        if text:
            self.text_entry.delete(0, tk.END)
            self.add_message(text, "user")
            self.process_query(text)
    
    def process_query(self, query):
        try:
            # Set processing status
            self.status_var.set("Processing...")
            self.root.update()
            
            # Process the query
            response = self.assistant.process_query(query)
            
            # Add response to chat
            self.add_message(response, "assistant")
            
            # Speak the response
            threading.Thread(
                target=self.assistant.speak,
                args=(response, self.assistant.current_language),
                daemon=True
            ).start()
            
            # Update status
            self.status_var.set("Ready")
        except Exception as e:
            self.logger.log(f"Error processing query: {e}", "ERROR")
            self.status_var.set("Error: Could not process query")
            self.add_message("Sorry, I couldn't process that request.", "system")
    
    def add_message(self, message, role):
        self.chat_text.config(state="normal")
        
        if message:
            # Add timestamp
            timestamp = time.strftime("%H:%M:%S")
            
            if role == "user":
                self.chat_text.insert(tk.END, f"[{timestamp}] You: ", "user_prefix")
                self.chat_text.insert(tk.END, f"{message}\n\n", "user_message")
            elif role == "assistant":
                name = self.assistant.active_personality.name
                avatar = self.assistant.active_personality.avatar
                self.chat_text.insert(tk.END, f"[{timestamp}] {avatar} {name}: ", "assistant_prefix")
                self.chat_text.insert(tk.END, f"{message}\n\n", "assistant_message")
            else:  # System message
                self.chat_text.insert(tk.END, f"[{timestamp}] System: ", "system_prefix")
                self.chat_text.insert(tk.END, f"{message}\n\n", "system_message")
            
            # Configure tags
            self.chat_text.tag_configure("user_prefix", foreground="#007bff", font=(Config.CHAT_FONT[0], Config.CHAT_FONT[1], "bold"))
            self.chat_text.tag_configure("assistant_prefix", foreground="#28a745", font=(Config.CHAT_FONT[0], Config.CHAT_FONT[1], "bold"))
            self.chat_text.tag_configure("system_prefix", foreground="#dc3545", font=(Config.CHAT_FONT[0], Config.CHAT_FONT[1], "bold"))
            
            # Scroll to bottom
            self.chat_text.see(tk.END)
        
        self.chat_text.config(state="disabled")
    
    def clear_chat(self):
        self.chat_text.config(state="normal")
        self.chat_text.delete(1.0, tk.END)
        self.chat_text.config(state="disabled")
        self.status_var.set("Chat cleared")
    
    def on_closing(self):
        try:
            if self.assistant.is_listening:
                self.assistant.stop_listening()
                
            if self.listening_thread and self.listening_thread.is_alive():
                # Give the thread time to clean up
                self.listening_thread.join(timeout=1.0)
            
            self.root.destroy()
        except Exception as e:
            self.logger.log(f"Error during closing: {e}", "ERROR")
            self.root.destroy()

# Main application entry point
def main():
    try:
        # Create main window
        root = tk.Tk()
        
        # Initialize voice assistant
        assistant = VoiceAssistant()
        
        # Create GUI
        app = ModernAssistantGUI(root, assistant)
        
        # Add initial system message
        app.add_message(f"Welcome to {Config.APP_NAME} v{Config.VERSION}! I'm your AI voice assistant.", "system")
        app.add_message(f"I'm currently using the personality of {assistant.active_personality.name}. Ask me anything!", "assistant")
        
        # Start the application
        root.mainloop()
    except Exception as e:
        Logger().log(f"Fatal error: {e}", "CRITICAL")
        if "root" in locals() and root:
            messagebox.showerror("Error", f"A fatal error occurred:\n{e}")
            root.destroy()

if __name__ == "__main__":
    main()

