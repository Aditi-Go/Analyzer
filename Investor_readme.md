# Founder-Investor Matching System

A smart tool that connects startup founders with potential investors based on compatibility scores.

## Features

- **AI-Powered Matching**: Utilizes Google's Gemini AI to analyze startup-investor compatibility
- **Smart Fallback System**: Continues to function even when API is unavailable
- **Interactive Interface**: Easy-to-use Streamlit web interface
- **Data Visualization**: Color-coded match scores and progress tracking
- **Flexible Data Import**: Upload your own JSON data or use sample data
- **Results Export**: Download matching results as CSV
- **Match Statistics**: View high, medium, and low matches at a glance

## How It Works

1. **Upload**: Submit your JSON data or use sample data
2. **Select**: Choose a founder from the dropdown menu
3. **Analysis**: We analyze each investor and calculate compatibility scores
4. **Results**: View color-coded match scores in a sortable table

## Technical Details

- Built with Python and Streamlit
- Uses Google's Gemini 2.0 Flash for match scoring
- Data processing with pandas
- Combines AI-based scoring with fallback rule-based matching

## Getting Started

### Prerequisites

- Python 3.7+
- Gemini API key

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
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Usage

1. Run the app:
   ```bash
   streamlit run investor_analyzer.py
   ```

2. Open your browser (typically http://localhost:8501)

3. Upload JSON data or use sample data

4. Select a founder and click "Find Matches"

## Key Components

### Data Format
The system expects JSON data with fields for:
- Name
- Industry
- Stage
- Cheque_range/funding_required
- Overview
- Type

### Matching Algorithm
Combines:
- AI-based scoring using Gemini
- Rule-based fallback system
- Field mapping for different entity types

## Understanding Results

- **Match Score**: Value between 0 and 1 (higher is better)
- **Color Coding**:
  - Green (>0.7): High match
  - Yellow (0.4-0.7): Medium match
  - Pink (<0.4): Low match

