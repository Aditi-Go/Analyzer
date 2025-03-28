# AI-Powered Startup Analysis Tools

This repository contains two powerful tools for startup ecosystem analysis:

## 1. Pitch Deck Analyzer
A smart tool that analyzes startup pitch decks to provide evaluations, summaries, and insights.

### Key Features
- Upload and analyze pitch deck PDFs
- Get detailed evaluations with section-by-section scoring
- Generate concise summaries of key components
- Interactive Q&A about pitch deck content
- Compare multiple pitch decks side by side

[Learn more about the Pitch Deck Analyzer](Pitch_readme.md)

## 2. Founder-Investor Matching System 
An intelligent system that connects startup founders with potential investors based on compatibility.

### Key Features
- AI-powered matching using Google's Gemini
- Smart fallback scoring system
- Interactive web interface
- Color-coded match visualization
- Flexible data import/export
- Match statistics and analytics

[Learn more about the Founder-Investor Matching System](Investor_readme.md)

## Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key (for Pitch Analyzer)
- Gemini API key (for Investor Matching)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Aditi-Go/Analyzer.git
   cd Analyzer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Running the Applications

1. For Pitch Deck Analyzer:
   ```bash
   streamlit run pitch_analyzer.py
   ```

2. For Founder-Investor Matching:
   ```bash
   streamlit run investor_analyzer.py
   ```

Both applications will be accessible through your browser at http://localhost:8501

