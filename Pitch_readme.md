# AI Pitch Deck Analyzer

A smart tool that analyzes startup pitch decks to provide evaluations, summaries, and insights.

## Features

- **Pitch Deck Analysis**: Upload PDFs and get detailed evaluations with scores
- **Section Scoring**: Scores for Problem, Solution, Market, Business Model, Financials, and Team
- **Summaries**: Get concise summaries of key pitch deck components
- **Q&A**: Ask questions about any pitch deck
- **Compare Decks**: View multiple pitch decks side by side

## How It Works

1. **Upload**: Submit your pitch deck PDF
2. **Analysis**: We extract and analyze the content using GPT-4
3. **Results**: View scores, summaries, and recommendations
4. **Insights**: Ask follow-up questions about the pitch

## Technical Details

- Built with Python and Streamlit
- Uses GPT-4 for text analysis
- PDF extraction with PyPDF2
- Data analysis with regex pattern matching

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/Aditi-Go/Analyzer.git]
   cd Analyzer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file and add your API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Usage

1. Run the app:
   ```bash
   streamlit run pitch_analyzer.py
   ```

2. Open your browser (typically http://localhost:8501)

3. Upload pitch deck PDFs using the file uploader

4. View results and interact with the tool

## Key Components

### PitchDeck Class
Stores pitch deck information:
- Company name
- Extracted text
- Scores
- Summary
- Evaluation details

### PitchDeckAnalyzer Class
Handles analysis:
- PDF text extraction
- Company name identification
- Evaluation
- Summary generation
- Q&A

## Analysis Methodology

We evaluate pitch decks across six dimensions:

1. **Problem (15%)**: Clarity and significance of the problem
2. **Solution (20%)**: Effectiveness and uniqueness of the solution
3. **Market (15%)**: Market size, growth potential, and target audience
4. **Business Model (15%)**: Revenue strategy and go-to-market approach
5. **Financials (20%)**: Projections, metrics, and investment requirements
6. **Team (15%)**: Experience, expertise, and execution ability

