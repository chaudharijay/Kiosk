import streamlit as st
import pandas as pd
import os
from datetime import datetime
import tempfile
import io
import re
import difflib

# Try to import speech recognition libraries
try:
    from audio_recorder_streamlit import audio_recorder
    import speech_recognition as sr
    BASIC_AUDIO_ENABLED = True
except ImportError:
    BASIC_AUDIO_ENABLED = False

# Try to import Whisper for better transcription
try:
    import whisper
    WHISPER_ENABLED = True
except ImportError:
    WHISPER_ENABLED = False

# Try to import pydub for audio conversion
try:
    from pydub import AudioSegment
    PYDUB_ENABLED = True
except ImportError:
    PYDUB_ENABLED = False

# Page configuration with minimal interface
st.set_page_config(
    page_title="Registration System",
    page_icon="🎤",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Professional CSS with clean white theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
        line-height: 1.3;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .answer-display {
        font-size: 1.1rem;
        color: #27ae60;
        margin: 1.5rem 0;
        padding: 1.2rem;
        background-color: #f8f9fa;
        border-radius: 6px;
        border-left: 4px solid #27ae60;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .error-display {
        font-size: 1.1rem;
        color: #dc3545;
        margin: 1.5rem 0;
        padding: 1.2rem;
        background-color: #f8f9fa;
        border-radius: 6px;
        border-left: 4px solid #dc3545;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-display {
        font-size: 1.1rem;
        color: #fd7e14;
        margin: 1.5rem 0;
        padding: 1.2rem;
        background-color: #f8f9fa;
        border-radius: 6px;
        border-left: 4px solid #fd7e14;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .keyword-match {
        font-size: 1.1rem;
        color: #495057;
        margin: 1rem 0;
        padding: 1.2rem;
        background-color: #f8f9fa;
        border-radius: 6px;
        border-left: 4px solid #6c757d;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .suggestion-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        border-left: 4px solid #6c757d;
        font-size: 0.95rem;
        color: #495057;
    }
    .progress-text {
        font-size: 1rem;
        text-align: center;
        color: #6c757d;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .stButton > button {
        font-size: 1rem !important;
        padding: 0.7rem 1.8rem !important;
        border-radius: 6px !important;
        border: 1px solid #dee2e6 !important;
        background-color: #ffffff !important;
        color: #495057 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #f8f9fa !important;
        border-color: #adb5bd !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
    }
    .success-message {
        background-color: #f8f9fa;
        color: #28a745;
        padding: 1.5rem;
        border-radius: 6px;
        border-left: 4px solid #28a745;
        text-align: center;
        margin: 2rem 0;
        font-size: 1.1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .audio-container {
        margin: 2rem 0;
        text-align: center;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    .mic-wrapper {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        margin: 1rem 0 !important;
    }
    .mic-wrapper > div {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin: 0 auto !important;
    }
    .record-instruction {
        font-size: 1.1rem;
        color: #495057;
        margin-top: 1rem;
        font-weight: 400;
    }
    .divider {
        margin: 2rem 0;
        border-top: 1px solid #dee2e6;
    }
    .summary-title {
        font-size: 1.8rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    .summary-item {
        background-color: #ffffff;
        padding: 1.2rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .summary-question {
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    .summary-answer {
        color: #495057;
        font-size: 1.05rem;
        line-height: 1.4;
    }

</style>
""", unsafe_allow_html=True)

# Keyword recognition data
COUNTRIES = [
    'Afghanistan', 'Albania', 'Algeria', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan',
    'Bahrain', 'Bangladesh', 'Belarus', 'Belgium', 'Brazil', 'Bulgaria', 'Cambodia', 'Canada', 'Chile',
    'China', 'Colombia', 'Croatia', 'Czech Republic', 'Denmark', 'Egypt', 'Estonia', 'Finland', 'France',
    'Georgia', 'Germany', 'Greece', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland',
    'Israel', 'Italy', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kuwait', 'Latvia', 'Lebanon', 'Lithuania',
    'Luxembourg', 'Malaysia', 'Mexico', 'Morocco', 'Netherlands', 'New Zealand', 'Nigeria', 'Norway',
    'Pakistan', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'Saudi Arabia',
    'Singapore', 'Slovakia', 'Slovenia', 'South Africa', 'South Korea', 'Spain', 'Sri Lanka', 'Sweden',
    'Switzerland', 'Thailand', 'Turkey', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States',
    'USA', 'UK', 'UAE', 'Vietnam'
]

COMMON_NAME_PATTERNS = [
    r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
    r'\b[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+\b',  # First M. Last
    r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Middle Last
]

COMPANY_KEYWORDS = [
    'company', 'corporation', 'corp', 'inc', 'ltd', 'limited', 'llc', 'group', 'enterprises',
    'solutions', 'systems', 'technologies', 'tech', 'software', 'services', 'consulting',
    'industries', 'international', 'global', 'associates', 'partners', 'holdings'
]

PRODUCT_KEYWORDS = [
    'software', 'hardware', 'application', 'app', 'platform', 'system', 'tool', 'service',
    'solution', 'product', 'technology', 'device', 'equipment', 'machine', 'instrument',
    'computer', 'laptop', 'phone', 'tablet', 'cloud', 'database', 'analytics', 'ai',
    'artificial intelligence', 'machine learning', 'ml', 'automation', 'crm', 'erp'
]

# Pharmaceutical and Medical Keywords
PHARMACEUTICAL_KEYWORDS = [
    'pivmecillinam', 'antibiotic', 'antibiotics', 'penicillin', 'amoxicillin', 'drug', 'drugs',
    'medicine', 'medication', 'pharmaceutical', 'pharma', 'vaccine', 'vaccines', 'therapy',
    'treatment', 'clinical', 'medical', 'healthcare', 'health', 'biotechnology', 'biotech',
    'diagnostic', 'therapeutics', 'oncology', 'cardiology', 'neurology', 'insulin', 'generic',
    'branded', 'formulation', 'tablet', 'capsule', 'injection', 'syrup', 'ointment', 'cream',
    'prescription', 'otc', 'over the counter', 'research', 'development', 'trial', 'study',
    'aspirin', 'ibuprofen', 'paracetamol', 'metformin', 'atorvastatin', 'amlodipine', 'omeprazole',
    'losartan', 'hydrochlorothiazide', 'levothyroxine', 'simvastatin', 'lisinopril', 'warfarin',
    'clopidogrel', 'rosuvastatin', 'ramipril', 'bisoprolol', 'furosemide', 'prednisolone'
]



JOB_TITLES = [
    'manager', 'director', 'engineer', 'developer', 'analyst', 'consultant', 'specialist',
    'coordinator', 'supervisor', 'executive', 'officer', 'president', 'ceo', 'cto', 'cfo',
    'vice president', 'vp', 'senior', 'junior', 'lead', 'head', 'chief', 'assistant',
    'associate', 'administrator', 'representative', 'sales', 'marketing', 'hr', 'it',
    'operations', 'finance', 'technical', 'business', 'project'
]

# Initialize session state
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'responses' not in st.session_state:
    st.session_state.responses = {}
if 'registration_complete' not in st.session_state:
    st.session_state.registration_complete = False
if 'data_saved' not in st.session_state:
    st.session_state.data_saved = False
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = None

# Questions and keys
QUESTIONS = [
    "Please tell us your name?",
    "What is your designation/job title?", 
    "Name of your organisation",
    "Country your company belongs from?",
    "What product are you interested in?"
]

QUESTION_KEYS = [
    "name",
    "designation", 
    "organisation",
    "country",
    "product_interest"
]

EXCEL_FILENAME = f"conference_registrations_{datetime.now().strftime('%Y%m%d')}.xlsx"

@st.cache_resource
def load_whisper_model():
    """Load Whisper model (cached for performance)"""
    if WHISPER_ENABLED:
        try:
            return whisper.load_model("base")
        except Exception as e:
            st.error(f"Error loading Whisper model: {e}")
            return None
    return None

def find_closest_match(text, word_list, threshold=0.6):
    """Find closest match from a word list using fuzzy matching"""
    best_match = None
    best_score = 0
    
    for word in word_list:
        similarity = difflib.SequenceMatcher(None, text.lower(), word.lower()).ratio()
        if similarity > best_score and similarity >= threshold:
            best_score = similarity
            best_match = word
    
    return best_match, best_score

def extract_keywords(text, keywords):
    """Extract keywords from text"""
    found_keywords = []
    text_lower = text.lower()
    
    for keyword in keywords:
        if keyword.lower() in text_lower:
            found_keywords.append(keyword)
    
    return found_keywords

def extract_name(text):
    """Extract clean name from natural language response"""
    text = text.strip()
    
    # Remove common prefixes
    prefixes = [
        r"my name is\s+", r"i am\s+", r"this is\s+", r"i'm\s+", 
        r"the name is\s+", r"it's\s+", r"name\s*:\s*", r"i am called\s+"
    ]
    
    for prefix in prefixes:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE)
    
    # Remove articles if they appear (rare for names but possible)
    text = re.sub(r"^(a|an|the)\s+", "", text, flags=re.IGNORECASE)
    
    # Extract name patterns
    for pattern in COMMON_NAME_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group().strip()
    
    # If no pattern matches, return cleaned text
    words = text.split()
    if len(words) <= 3 and all(word.replace(".", "").replace("-", "").replace("'", "").isalpha() for word in words):
        return text.strip()
    
    return text.strip()

def extract_designation(text):
    """Extract job title from natural language response"""
    text = text.strip()
    
    # Remove common prefixes
    prefixes = [
        r"i work as\s+", r"i am a\s+", r"i'm a\s+", r"my designation is\s+",
        r"my job title is\s+", r"i am an\s+", r"i'm an\s+", r"my role is\s+",
        r"designation\s*:\s*", r"job title\s*:\s*", r"position\s*:\s*"
    ]
    
    for prefix in prefixes:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE)
    
    # Remove articles at the beginning
    text = re.sub(r"^(a|an|the)\s+", "", text, flags=re.IGNORECASE)
    
    # Look for job title keywords and extract surrounding context
    for title in JOB_TITLES:
        if title.lower() in text.lower():
            # Find the job title with context
            pattern = rf"\b[\w\s]*{re.escape(title)}\b[\w\s]*"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result = match.group().strip()
                # Remove articles from the result too
                result = re.sub(r"^(a|an|the)\s+", "", result, flags=re.IGNORECASE)
                return result
    
    return text.strip()

def extract_organisation(text):
    """Extract organization name from natural language response"""
    text = text.strip()
    
    # Remove common prefixes first (ordered from most specific to least specific)
    prefixes = [
        r"the name of my organisation is\s+", r"the name of my organization is\s+",
        r"the name of my company is\s+", r"the name of the organisation is\s+",
        r"the name of the organization is\s+", r"the name of the company is\s+",
        r"name of my organisation is\s+", r"name of my organization is\s+",
        r"name of my company is\s+", r"name of the organisation is\s+",
        r"name of the organization is\s+", r"name of the company is\s+",
        r"my organisation name is\s+", r"my organization name is\s+",
        r"my company name is\s+", r"our company name is\s+",
        r"our organisation name is\s+", r"our organization name is\s+",
        r"organisation name is\s+", r"organization name is\s+",
        r"company name is\s+", r"the company name is\s+",
        r"i work at\s+", r"i work for\s+", r"i am from\s+", r"i'm from\s+",
        r"my company is\s+", r"our company is\s+", r"the company is\s+",
        r"my organisation is\s+", r"my organization is\s+", 
        r"our organisation is\s+", r"our organization is\s+",
        r"organisation\s*:\s*", r"organization\s*:\s*", r"company\s*:\s*", 
        r"we are\s+", r"it is\s+", r"i work in\s+", r"working at\s+", 
        r"working for\s+", r"employed at\s+", r"from\s+", r"at\s+"
    ]
    
    for prefix in prefixes:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE)
        text = text.strip()  # Clean up after each removal
    
    # Remove articles at the beginning
    text = re.sub(r"^(a|an|the)\s+", "", text, flags=re.IGNORECASE)
    
    # Remove common unwanted words/phrases that might appear
    unwanted_words = [
        r"\bcalled\b", r"\bnamed\b", r"\bis\b", r"\bwas\b", r"\bwere\b",
        r"\bhere\b", r"\bthere\b", r"\bthis\b", r"\bthat\b"
    ]
    
    for unwanted in unwanted_words:
        text = re.sub(unwanted, "", text, flags=re.IGNORECASE)
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # If text is empty after cleaning, return original
    if not text:
        return text
    
    # Capitalize first letter of each word for consistency
    text = ' '.join(word.capitalize() for word in text.split())
    
    return text

def extract_country(text):
    """Extract country name from natural language response"""
    text = text.strip()
    
    # Remove common prefixes
    prefixes = [
        r"we are from\s+", r"i am from\s+", r"i'm from\s+", r"from\s+",
        r"our company is from\s+", r"located in\s+", r"based in\s+",
        r"country\s*:\s*", r"we belong to\s+", r"it's from\s+"
    ]
    
    for prefix in prefixes:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE)
    
    # Direct country match
    for country in COUNTRIES:
        if country.lower() in text.lower():
            return country
    
    # Fuzzy match for typos
    closest_match, score = find_closest_match(text, COUNTRIES, threshold=0.7)
    if closest_match and score > 0.8:
        return closest_match
    
    return text.strip()

def extract_product(text):
    """Extract pharmaceutical product name from natural language response"""
    text = text.strip()
    
    # Remove common prefixes
    prefixes = [
        r"i am interested in\s+", r"i'm interested in\s+", r"interested in\s+",
        r"we need\s+", r"we want\s+", r"looking for\s+", r"we are looking for\s+",
        r"product\s*:\s*", r"we use\s+", r"i need\s+", r"i want\s+",
        r"we are interested in\s+", r"our interest is in\s+"
    ]
    
    for prefix in prefixes:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE)
    
    # Remove articles at the beginning
    text = re.sub(r"^(a|an|the)\s+", "", text, flags=re.IGNORECASE)
    
    # Check for pivmecillinam specifically (important pharmaceutical product)
    if 'pivmecillinam' in text.lower():
        return 'Pivmecillinam'
    
    # Enhanced pharmaceutical product recognition
    # Check for common pharmaceutical products by name
    common_pharma_products = [
        'amoxicillin', 'penicillin', 'atorvastatin', 'metformin', 'omeprazole',
        'ibuprofen', 'aspirin', 'insulin', 'lisinopril', 'simvastatin',
        'amlodipine', 'hydrochlorothiazide', 'levothyroxine', 'warfarin',
        'clopidogrel', 'rosuvastatin', 'ramipril', 'bisoprolol', 'furosemide',
        'prednisolone', 'paracetamol', 'acetaminophen', 'losartan'
    ]
    
    for product in common_pharma_products:
        if product.lower() in text.lower():
            return product.capitalize()
    
    # Look for pharmaceutical categories and terms
    pharma_categories = {
        'antibiotic': 'Antibiotics',
        'antibiotics': 'Antibiotics', 
        'vaccine': 'Vaccines',
        'vaccines': 'Vaccines',
        'insulin': 'Insulin',
        'generic': 'Generic medicines',
        'pain relief': 'Pain relief medications',
        'cholesterol': 'Cholesterol medications',
        'diabetes': 'Diabetes medications',
        'blood pressure': 'Blood pressure medications',
        'hypertension': 'Hypertension medications',
        'oncology': 'Oncology treatments',
        'cancer': 'Cancer treatments',
        'cardiology': 'Cardiovascular medications',
        'heart': 'Cardiovascular medications'
    }
    
    text_lower = text.lower()
    for term, category in pharma_categories.items():
        if term in text_lower:
            return category
    
    # Look for pharmaceutical keywords with context
    all_keywords = PHARMACEUTICAL_KEYWORDS + PRODUCT_KEYWORDS
    for keyword in all_keywords:
        if keyword.lower() in text.lower():
            # Try to extract the product name with context
            words = text.split()
            for i, word in enumerate(words):
                if keyword.lower() in word.lower():
                    # Get 1-2 words before and after if they seem related
                    start = max(0, i-1)
                    end = min(len(words), i+2)
                    potential_product = " ".join(words[start:end])
                    if len(potential_product.split()) <= 3:
                        # Remove articles from result
                        potential_product = re.sub(r"^(a|an|the)\s+", "", potential_product, flags=re.IGNORECASE)
                        return potential_product.strip()
    
    # If nothing specific found, return cleaned text
    return text.strip()

def analyze_response(question_key, response_text):
    """Extract clean answer based on question type"""
    extractors = {
        'name': extract_name,
        'designation': extract_designation,
        'organisation': extract_organisation,
        'country': extract_country,
        'product_interest': extract_product
    }
    
    if question_key in extractors:
        extracted_answer = extractors[question_key](response_text)
        return extracted_answer
    
    return response_text.strip()

def get_suggestions(question_key, response_text):
    """Get suggestions based on question type and response"""
    suggestions = []
    
    if question_key == 'country':
        # Suggest similar countries
        matches = []
        for country in COUNTRIES:
            similarity = difflib.SequenceMatcher(None, response_text.lower(), country.lower()).ratio()
            if 0.3 <= similarity < 0.7:
                matches.append((country, similarity))
        
        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            suggestions = [f"Did you mean: {match[0]}?" for match in matches[:3]]
    
    elif question_key == 'designation':
        suggestions = ["Examples: Manager, Engineer, Director, Analyst, Developer, Sales Representative"]
    
    elif question_key == 'product_interest':
        suggestions = []
        
    return suggestions



def transcribe_with_whisper(audio_bytes):
    """Transcribe audio using OpenAI Whisper (high accuracy)"""
    try:
        if st.session_state.whisper_model is None:
            st.session_state.whisper_model = load_whisper_model()
        
        if st.session_state.whisper_model is None:
            return "Whisper model not available", True
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        result = st.session_state.whisper_model.transcribe(tmp_file_path)
        os.unlink(tmp_file_path)
        
        return result["text"].strip(), False
        
    except Exception as e:
        return f"Whisper transcription error: {str(e)}", True

def transcribe_with_google(audio_bytes):
    """Transcribe audio using Google Speech Recognition"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(tmp_file_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)
        
        os.unlink(tmp_file_path)
        text = recognizer.recognize_google(audio, language='en-US')
        return text, False
        
    except sr.UnknownValueError:
        return "Could not understand the audio. Please speak clearly and try again.", True
    except sr.RequestError as e:
        return f"Google Speech Recognition error: {e}", True
    except Exception as e:
        return f"Transcription error: {str(e)}", True

def save_to_excel_file():
    """Save registration data to Excel file (append if exists)"""
    try:
        new_data = {
            'Date': datetime.now().strftime("%Y-%m-%d"),
            'Time': datetime.now().strftime("%H:%M:%S"),
            'Name': st.session_state.responses.get('name', ''),
            'Designation': st.session_state.responses.get('designation', ''),
            'Organisation': st.session_state.responses.get('organisation', ''),
            'Country': st.session_state.responses.get('country', ''),
            'Product Interest': st.session_state.responses.get('product_interest', '')
        }
        
        new_df = pd.DataFrame([new_data])
        
        possible_filenames = [
            EXCEL_FILENAME,
            os.path.join(tempfile.gettempdir(), EXCEL_FILENAME),
            os.path.join(os.path.expanduser("~"), "Desktop", EXCEL_FILENAME),
            os.path.join(os.path.expanduser("~"), "Downloads", EXCEL_FILENAME)
        ]
        
        saved_successfully = False
        final_filename = None
        
        for filename in possible_filenames:
            try:
                if os.path.exists(filename):
                    existing_df = pd.read_excel(filename)
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    combined_df = new_df
                
                combined_df.to_excel(filename, index=False)
                saved_successfully = True
                final_filename = filename
                break
                
            except PermissionError:
                continue
            except Exception:
                continue
        
        if saved_successfully:
            return True, len(combined_df), final_filename
        else:
            output = io.BytesIO()
            if os.path.exists(EXCEL_FILENAME):
                try:
                    existing_df = pd.read_excel(EXCEL_FILENAME)
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                except:
                    combined_df = new_df
            else:
                combined_df = new_df
                
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                combined_df.to_excel(writer, sheet_name='Registrations', index=False)
            
            st.download_button(
                label="Download Registration Data",
                data=output.getvalue(),
                file_name=EXCEL_FILENAME,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            return True, len(combined_df), "Download"
        
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")
        return False, 0, None

def reset_registration():
    """Reset the registration process"""
    st.session_state.current_question = 0
    st.session_state.responses = {}
    st.session_state.registration_complete = False
    st.session_state.data_saved = False

def main():
    # Display TechQube logo in top-left corner
    col_logo, col_spacer = st.columns([1, 4])
    with col_logo:
        try:
            # Try to load and display the logo
            st.image("data:image/png;base64,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", width=120)
        except:
            # Fallback if logo can't be loaded
            st.markdown('<div style="color: #0f766e; font-weight: bold; font-size: 1.2rem;">TechQube AI Solutions</div>', unsafe_allow_html=True)
    
    if not (BASIC_AUDIO_ENABLED or WHISPER_ENABLED):
        st.error("""
        **No speech recognition libraries installed!**
        
        Install one of these options:
        
        **Option 1 (Recommended - Best Accuracy):**
        ```bash
        pip install openai-whisper
        ```
        
        **Option 2 (Basic):**
        ```bash
        pip install SpeechRecognition audio-recorder-streamlit
        ```
        """)
        return
    
    if not st.session_state.registration_complete:
        progress = st.session_state.current_question / len(QUESTIONS)
        st.progress(progress)
        st.markdown(f'<p class="progress-text">Question {st.session_state.current_question + 1} of {len(QUESTIONS)}</p>', 
                   unsafe_allow_html=True)
        
        current_q_index = st.session_state.current_question
        current_question = QUESTIONS[current_q_index]
        current_key = QUESTION_KEYS[current_q_index]
        
        st.markdown(f'<h1 class="main-header">{current_question}</h1>', unsafe_allow_html=True)
        
        if BASIC_AUDIO_ENABLED or WHISPER_ENABLED:
            st.markdown('<div class="audio-container">', unsafe_allow_html=True)
            st.markdown('<div class="mic-wrapper">', unsafe_allow_html=True)
            
            audio_bytes = audio_recorder(
                text="",
                recording_color="#44A08D",
                neutral_color="#7dd3fc",
                icon_name="microphone",
                icon_size="3x",
                pause_threshold=2.0,
                sample_rate=16000,
                key=f"audio_recorder_{current_q_index}"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<p class="record-instruction">Click to record your response</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if audio_bytes:
                transcription_method = "whisper" if WHISPER_ENABLED else "google"
                
                with st.spinner("Processing audio..."):
                    if transcription_method == "whisper":
                        transcription, is_error = transcribe_with_whisper(audio_bytes)
                    else:
                        transcription, is_error = transcribe_with_google(audio_bytes)
                
                if is_error:
                    st.markdown(f'<div class="error-display">{transcription}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="answer-display">Recorded: "{transcription}"</div>', unsafe_allow_html=True)
                    
                    # Extract clean answer from the transcription
                    extracted_answer = analyze_response(current_key, transcription)
                    
                    # Show extracted answer if different from original
                    if extracted_answer != transcription:
                        st.markdown(f'<div class="keyword-match">Extracted: "{extracted_answer}"</div>', unsafe_allow_html=True)
                    
                    edited_text = st.text_input("Edit if needed:", value=extracted_answer, key=f"edit_text_{current_q_index}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Record Again", use_container_width=True, key=f"record_again_{current_q_index}"):
                            st.rerun()
                    
                    with col2:
                        if st.button("Next Question", use_container_width=True, key=f"next_question_{current_q_index}"):
                            final_answer = edited_text if edited_text else extracted_answer
                            st.session_state.responses[current_key] = final_answer
                            st.session_state.current_question += 1
                            if st.session_state.current_question >= len(QUESTIONS):
                                st.session_state.registration_complete = True
                            st.rerun()
        
        # Fallback text input with extraction
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("**Or type your answer:**")
        manual_input = st.text_area("Type here:", height=80, key=f"manual_input_{current_q_index}")
        
        # Show extracted answer for manual input
        if manual_input.strip():
            extracted_manual = analyze_response(current_key, manual_input)
            if extracted_manual != manual_input.strip():
                st.markdown(f'<div class="keyword-match">Will save as: "{extracted_manual}"</div>', unsafe_allow_html=True)
        
        if st.button("Use Typed Answer", use_container_width=True, key=f"use_typed_{current_q_index}"):
            if manual_input.strip():
                extracted_answer = analyze_response(current_key, manual_input)
                st.session_state.responses[current_key] = extracted_answer
                st.session_state.current_question += 1
                if st.session_state.current_question >= len(QUESTIONS):
                    st.session_state.registration_complete = True
                st.rerun()
            else:
                st.error("Please provide an answer.")
        
        # Show completed responses
        if st.session_state.responses:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("**Previous Responses:**")
            for key, response in st.session_state.responses.items():
                question_index = QUESTION_KEYS.index(key)
                if question_index < current_q_index:
                    st.markdown(f'''
                    <div class="summary-item">
                        <div class="summary-question">{QUESTIONS[question_index]}</div>
                        <div class="summary-answer">{response}</div>
                    </div>
                    ''', unsafe_allow_html=True)
    
    else:
        if not st.session_state.data_saved:
            with st.spinner("Saving registration data..."):
                result = save_to_excel_file()
                if len(result) == 3:
                    success, total_records, save_location = result
                else:
                    success, total_records = result
                    save_location = EXCEL_FILENAME
                    
                if success:
                    st.session_state.data_saved = True
                    if save_location == "Download":
                        st.markdown(f'<div class="success-message">Registration completed successfully!<br>Use the download button above to get the Excel file<br>Total registrations: {total_records}</div>', 
                                   unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="success-message">Registration completed successfully!<br>Data saved to: {save_location}<br>Total registrations: {total_records}</div>', 
                                   unsafe_allow_html=True)
                else:
                    st.error("Failed to save registration data.")
        
        st.markdown('<h2 class="summary-title">Registration Summary</h2>', unsafe_allow_html=True)
        
        for key, response in st.session_state.responses.items():
            question_index = QUESTION_KEYS.index(key)
            st.markdown(f'''
            <div class="summary-item">
                <div class="summary-question">{QUESTIONS[question_index]}</div>
                <div class="summary-answer">{response}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        if st.button("New Registration", use_container_width=True):
            reset_registration()
            st.rerun()

if __name__ == "__main__":
    main()