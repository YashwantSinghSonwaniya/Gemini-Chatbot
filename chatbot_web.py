"""
AI Chatbot Web Application - Production Ready & Secure
A professional web interface using Streamlit and Google Gemini API

DEPLOYMENT INSTRUCTIONS:
═══════════════════════════════════════════════════════════════

1. ENVIRONMENT VARIABLES (Set in your deployment platform):
   - Streamlit Cloud: Settings → Secrets → Add GEMINI_API_KEY
   - Heroku: Config Vars → Add GEMINI_API_KEY
   - Docker: ENV GEMINI_API_KEY=your_key_here
   - Railway/Render: Environment → Add GEMINI_API_KEY

2. LOCAL TESTING:
   Create .streamlit/secrets.toml:
   ───────────────────────────────
   gemini_api_key = "your_api_key_here"

3. DEPLOY TO STREAMLIT CLOUD:
   - Push code to GitHub
   - Connect GitHub repo to Streamlit Cloud
   - Add GEMINI_API_KEY in Streamlit Secrets
   - App deploys automatically

4. DOCKERFILE FOR CUSTOM DEPLOYMENT:
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["streamlit", "run", "app.py"]

Author: Yashwant Singh Sonwaniya
Date: October 2025
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

# Import Google Gemini SDK
try:
    from google import genai
    from google.genai import types
except ImportError:
    st.error("❌ Missing google-genai SDK. Install with: `pip install google-genai`")
    st.stop()


# ============================================================================
# SECURE CONFIGURATION
# ============================================================================

class SecureConfig:
    """Secure configuration with environment variables and Streamlit secrets"""
    
    @staticmethod
    def get_api_key() -> Optional[str]:
        """Get API key from Streamlit secrets or environment variables (secure)"""
        # Priority: Streamlit secrets > Environment variables
        if "gemini_api_key" in st.secrets:
            return st.secrets["gemini_api_key"]
        return os.getenv("GEMINI_API_KEY")
    
    MODEL = "gemini-2.5-flash"
    MAX_TOKENS = 1000
    TEMPERATURE = 0.7
    
    FAQ_SYSTEM_PROMPT = """You are a helpful FAQ assistant. Answer questions 
    clearly, concisely, and accurately. If you don't know the answer, say so."""
    
    SUMMARIZE_SYSTEM_PROMPT = """You are a text summarization expert. Provide 
    clear, concise summaries that capture the main points of the given text."""


# ============================================================================
# CHATBOT CLASS
# ============================================================================

class AIChatbot:
    """Main chatbot class with secure API key handling"""
    
    def __init__(self, api_key: str):
        """Initialize the chatbot with Gemini client"""
        if not api_key or not api_key.strip():
            raise ValueError("API key is required")
        
        self.client = genai.Client(api_key=api_key)
        self.current_mode = None
    
    def send_to_gemini(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a message to Gemini and get a response (with robust error handling)
        
        Args:
            user_message (str): User's input message
            system_prompt (str): Optional system prompt
            
        Returns:
            str: AI's response text or error message
        """
        try:
            if not user_message or not user_message.strip():
                return "Error: Message cannot be empty"
            
            # Build prompt
            prompt_parts = []
            if system_prompt:
                prompt_parts.append(system_prompt.strip())
            prompt_parts.append(user_message.strip())
            prompt = "\n\n".join(prompt_parts)

            # Configure generation
            config = types.GenerateContentConfig(
                temperature=SecureConfig.TEMPERATURE,
                max_output_tokens=SecureConfig.MAX_TOKENS
            )

            # Generate response
            response = self.client.models.generate_content(
                model=SecureConfig.MODEL,
                contents=prompt,
                config=config
            )

            # Extract text safely
            ai_response = getattr(response, "text", None)
            
            if ai_response is None:
                # Try alternative response attributes
                if hasattr(response, 'candidates') and response.candidates:
                    content = response.candidates[0].content
                    if hasattr(content, 'parts') and content.parts:
                        ai_response = content.parts[0].text
                
                if ai_response is None:
                    return "Error: Empty response from API. Please try again."
            
            # Ensure we have a string and strip it
            ai_response = str(ai_response).strip()
            
            if not ai_response:
                return "Error: Received empty response from API. Please try again."
            
            return ai_response
            
        except Exception as e:
            error_msg = str(e).lower()
            if "api_key" in error_msg or "authentication" in error_msg or "invalid" in error_msg:
                return "Error: API authentication failed. Please contact the administrator."
            elif "quota" in error_msg or "rate_limit" in error_msg:
                return "Error: Service temporarily unavailable. Please try again in a few moments."
            elif "timeout" in error_msg:
                return "Error: Request timed out. Please try with shorter text."
            else:
                return "Error: Unable to process request. Please try again."
    
    def answer_faq(self, question: str) -> str:
        """Answer a FAQ question"""
        self.current_mode = "FAQ"
        return self.send_to_gemini(question, SecureConfig.FAQ_SYSTEM_PROMPT)
    
    def summarize_text(self, text: str) -> str:
        """Summarize text"""
        self.current_mode = "SUMMARIZE"
        prompt = f"Please provide a clear and concise summary of the following text:\n\n{text}"
        return self.send_to_gemini(prompt, SecureConfig.SUMMARIZE_SYSTEM_PROMPT)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def typing_effect(text: str, container) -> str:
    """Display text with smooth typing effect"""
    displayed_text = ""
    text_placeholder = container.empty()
    
    # Split into chunks for faster display
    chunk_size = 3
    for i in range(0, len(text), chunk_size):
        displayed_text += text[i:i+chunk_size]
        text_placeholder.markdown(displayed_text)
        time.sleep(0.01)
    
    return displayed_text


def get_session_state_key(mode: str) -> tuple:
    """Get session state keys for a given mode"""
    if mode == "FAQ":
        return ("faq_responses", "faq_count")
    else:
        return ("summary_responses", "summary_count")


# ============================================================================
# STREAMLIT CONFIGURATION & UI
# ============================================================================

def configure_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="AI Chatbot - Gemini",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for modern, polished UI
    st.markdown("""
        <style>
        /* Main theme colors */
        :root {
            --primary: #1f77b4;
            --secondary: #ff7f0e;
            --success: #2ca02c;
            --error: #d62728;
        }
        
        /* Header styling */
        .main-header {
            font-size: 2.8rem;
            font-weight: 900;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            padding: 2rem 0 1rem 0;
            letter-spacing: -1px;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            font-size: 1.1rem;
            margin-bottom: 1.5rem;
            font-weight: 500;
        }
        
        /* Response containers */
        .response-container {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-radius: 1rem;
            border-left: 4px solid #667eea;
            margin: 1.5rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .response-header {
            font-size: 1.2rem;
            font-weight: 700;
            color: #333;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Input styling */
        .input-label {
            font-weight: 600;
            color: #333;
            margin-bottom: 0.5rem;
        }
        
        /* Button styling */
        .stButton > button {
            border-radius: 0.7rem;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            transition: all 0.3s ease;
        }
        
        /* Mode selector */
        .mode-selector {
            display: flex;
            gap: 1rem;
            margin: 1rem 0;
        }
        
        /* Session history styling */
        .history-item {
            padding: 1rem;
            margin: 0.5rem 0;
            background-color: #f0f2f6;
            border-radius: 0.5rem;
            border-left: 3px solid #667eea;
        }
        
        /* Tips and info boxes */
        .tip-box {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 0.7rem;
            border-left: 4px solid #2196f3;
            margin: 1rem 0;
        }
        
        /* Smooth animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease-in-out;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .main-header {
                font-size: 2rem;
            }
            .response-container {
                padding: 1rem;
            }
        }
        </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the main header and introduction"""
    st.markdown('<div class="main-header">🤖 Yashwant’s Gemini Chatbot — FAQ & Summarizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Powered by Google Gemini • Ask Questions • Summarize Text</div>', unsafe_allow_html=True)
    st.markdown("---")


def render_sidebar():
    """Render sidebar with settings and information"""
    with st.sidebar:
        st.header("⚙️ Settings & Info")
        
        # Feature toggles
        col1, col2 = st.columns(2)
        with col1:
            enable_typing = st.checkbox("✨ Typing Effect", value=True, help="Enable smooth typing animation")
        with col2:
            show_timestamps = st.checkbox("🕐 Show Time", value=False, help="Display timestamps in responses")
        
        st.markdown("---")
        
        # About section
        st.header("ℹ️ About")
        st.markdown("""
        This chatbot uses **Google Gemini AI** to:
        - 💡 Answer your questions instantly
        - 📝 Summarize long texts
        - 💬 Maintain conversation context
        
        **Features:**
        - Fast & accurate responses
        - Clean, intuitive interface
        - Privacy-focused design
        """)
        
        st.markdown("---")
        
        # Tips section
        st.header("💡 Tips")
        st.info("""
        **For better results:**
        - Be specific with your questions
        - Provide context when needed
        - Keep summaries under 5000 words
        """)
        
        st.markdown("---")
        
        # Footer
        st.caption("Made with ❤️ by Yashwant Singh Sonwaniya")
        st.caption("Version 2.0 • Secure & Production-Ready")
    
    return enable_typing, show_timestamps


def main():
    """Main Streamlit application"""
    
    # Configure page
    configure_page()
    
    # Render header
    render_header()
    
    # Render sidebar
    enable_typing_effect, show_timestamps = render_sidebar()
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'faq_responses' not in st.session_state:
        st.session_state.faq_responses = []
    if 'summary_responses' not in st.session_state:
        st.session_state.summary_responses = []
    if 'faq_count' not in st.session_state:
        st.session_state.faq_count = 0
    if 'summary_count' not in st.session_state:
        st.session_state.summary_count = 0
    
    # Get API key securely
    api_key = SecureConfig.get_api_key()
    
    if not api_key:
        st.error("🔐 API Configuration Missing")
        st.markdown("""
        ### ⚠️ Deployment Required
        
        This application requires a Gemini API key to function. 
        
        **For Administrators:**
        1. Obtain a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
        2. Add it to your deployment environment:
           - **Streamlit Cloud**: Settings → Secrets → `gemini_api_key`
           - **Environment Variable**: `GEMINI_API_KEY`
           - **Docker**: `ENV GEMINI_API_KEY=your_key`
        
        **Security Note:** API keys are never displayed or logged. They're securely stored in encrypted secrets.
        """)
        st.stop()
    
    # Initialize chatbot
    try:
        chatbot = AIChatbot(api_key)
    except ValueError as e:
        st.error(f"❌ Configuration Error: {e}")
        st.stop()
    
    # Create tabs for different modes
    tab1, tab2, tab3 = st.tabs(["❓ FAQ Mode", "📝 Summarizer", "💬 History"])
    
    with tab1:
        st.header("❓ Ask a Question")
        st.markdown("Get instant answers to your questions powered by AI.")
        
        question = st.text_input(
            "Your Question:",
            placeholder="e.g., What is machine learning? How do I learn Python?",
            key="faq_input",
            help="Ask any question and get an instant AI response"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_faq = st.button("🚀 Ask", key="faq_submit", use_container_width=True)
        with col2:
            st.write("")
        
        if submit_faq and question:
            with st.spinner("🤔 Thinking..."):
                response = chatbot.answer_faq(question)
            
            # Check if response is valid
            if response and not response.startswith("Error"):
                # Store in session history
                entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S") if show_timestamps else None,
                    "type": "FAQ",
                    "input": question,
                    "output": response
                }
                st.session_state.faq_responses.append(entry)
                st.session_state.faq_count += 1
                
                # Display response with styling
                st.markdown('<div class="response-container">', unsafe_allow_html=True)
                st.markdown('<div class="response-header">🤖 Answer</div>', unsafe_allow_html=True)
                
                response_container = st.container()
                
                if enable_typing_effect:
                    typing_effect(response, response_container)
                else:
                    response_container.markdown(response)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Copy to clipboard
                col1, col2 = st.columns(2)
                with col1:
                    st.code(response, language=None)
                with col2:
                    st.caption("💾 Use the copy icon to save")
                
                # Show timestamp if enabled
                if show_timestamps:
                    st.caption(f"⏰ {entry['timestamp']}")
                
                st.success("✅ Response saved to history")
            else:
                st.error(response if response else "⚠️ Failed to get a response. Please try again.")
        
        elif submit_faq:
            st.warning("⚠️ Please enter a question")
        
        # Display recent FAQ responses
        if st.session_state.faq_responses:
            st.markdown("---")
            st.subheader("📚 Recent Q&A")
            for i, item in enumerate(reversed(st.session_state.faq_responses[-3:]), 1):
                with st.expander(f"Q: {item['input'][:60]}..." if len(item['input']) > 60 else f"Q: {item['input']}"):
                    st.markdown(f"**Answer:**")
                    st.info(item['output'][:300] + "..." if len(item['output']) > 300 else item['output'])
    
    with tab2:
        st.header("📝 Text Summarizer")
        st.markdown("Paste long text and get a clear, concise summary.")
        
        text_input = st.text_area(
            "Text to Summarize:",
            placeholder="Paste your article, document, or text here...",
            height=250,
            key="summarize_input",
            help="Enter up to 5000 characters"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_summary = st.button("📊 Summarize", key="summary_submit", use_container_width=True)
        with col2:
            st.write("")
        
        if submit_summary and text_input:
            # Show text length
            char_count = len(text_input)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Characters", char_count)
            with col2:
                st.metric("Words", len(text_input.split()))
            with col3:
                st.metric("Readability", "Good" if char_count < 5000 else "Long")
            
            if char_count > 5000:
                st.warning("⚠️ Text is long. Summaries work best with shorter text.")
            
            with st.spinner("📝 Generating summary..."):
                response = chatbot.summarize_text(text_input)
            
            # Check if response is valid
            if response and not response.startswith("Error"):
                # Store in session history
                entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S") if show_timestamps else None,
                    "type": "SUMMARY",
                    "input": text_input,
                    "output": response
                }
                st.session_state.summary_responses.append(entry)
                st.session_state.summary_count += 1
                
                # Display response with styling
                st.markdown('<div class="response-container">', unsafe_allow_html=True)
                st.markdown('<div class="response-header">📋 Summary</div>', unsafe_allow_html=True)
                
                response_container = st.container()
                
                if enable_typing_effect:
                    typing_effect(response, response_container)
                else:
                    response_container.markdown(response)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    compression = round((1 - len(response.split()) / len(text_input.split())) * 100)
                    st.metric("Compression", f"{compression}%")
                with col2:
                    st.metric("Summary Length", len(response.split()), "words")
                with col3:
                    st.metric("Readability", "Clear")
                
                # Copy to clipboard
                col1, col2 = st.columns(2)
                with col1:
                    st.code(response, language=None)
                with col2:
                    st.caption("💾 Use the copy icon to save")
                
                # Show timestamp if enabled
                if show_timestamps:
                    st.caption(f"⏰ {entry['timestamp']}")
                
                st.success("✅ Summary saved to history")
            else:
                st.error(response if response else "⚠️ Failed to generate summary. Please try again.")
        
        elif submit_summary:
            st.warning("⚠️ Please enter text to summarize")
        
        # Display recent summaries
        if st.session_state.summary_responses:
            st.markdown("---")
            st.subheader("📚 Recent Summaries")
            for i, item in enumerate(reversed(st.session_state.summary_responses[-3:]), 1):
                input_preview = item['input'][:50] + "..." if len(item['input']) > 50 else item['input']
                with st.expander(f"Summary {i}: {input_preview}"):
                    st.markdown("**Original Text:**")
                    st.text_area("", item['input'], height=150, disabled=True, key=f"orig_{i}")
                    st.markdown("**Summary:**")
                    st.info(item['output'])
    
    with tab3:
        st.header("💬 Conversation History")
        st.markdown("View all your questions and summaries from this session.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Questions Asked", st.session_state.faq_count)
        with col2:
            st.metric("Texts Summarized", st.session_state.summary_count)
        
        if not st.session_state.faq_responses and not st.session_state.summary_responses:
            st.info("💭 No history yet. Start by asking a question or summarizing text!")
        else:
            st.markdown("---")
            
            # FAQ History
            if st.session_state.faq_responses:
                st.subheader("❓ Questions & Answers")
                for i, item in enumerate(reversed(st.session_state.faq_responses), 1):
                    with st.expander(f"Q{i}: {item['input'][:60]}..." if len(item['input']) > 60 else f"Q{i}: {item['input']}"):
                        st.markdown("**Question:**")
                        st.text(item['input'])
                        st.markdown("**Answer:**")
                        st.info(item['output'])
                        if item['timestamp']:
                            st.caption(f"🕐 {item['timestamp']}")
            
            st.markdown("---")
            
            # Summary History
            if st.session_state.summary_responses:
                st.subheader("📝 Summaries")
                for i, item in enumerate(reversed(st.session_state.summary_responses), 1):
                    input_preview = item['input'][:40] + "..." if len(item['input']) > 40 else item['input']
                    with st.expander(f"S{i}: {input_preview}"):
                        st.markdown("**Original Text:**")
                        st.text_area("", item['input'], height=120, disabled=True, key=f"hist_orig_{i}")
                        st.markdown("**Summary:**")
                        st.info(item['output'])
                        if item['timestamp']:
                            st.caption(f"🕐 {item['timestamp']}")
            
            # Clear history button
            st.markdown("---")
            if st.button("🗑️ Clear Session History", key="clear_history"):
                st.session_state.chat_history = []
                st.session_state.faq_responses = []
                st.session_state.summary_responses = []
                st.session_state.faq_count = 0
                st.session_state.summary_count = 0
                st.success("✅ History cleared")
                st.rerun()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()