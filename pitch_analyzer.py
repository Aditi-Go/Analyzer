import openai
from dotenv import load_dotenv
import os
from typing import Optional, Dict, List, Tuple
import PyPDF2
import streamlit as st
from dataclasses import dataclass
from datetime import datetime
import json
import numpy as np
import re
import fitz
import io
from PIL import Image
import pytesseract
import tempfile
import base64
from io import BytesIO
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

@dataclass
class PitchDeck:
    id: str
    company_name: str
    deck_text: str
    summary: str = ""
    pitch_score: float = 0.0
    evaluation: str = ""
    section_scores: Dict[str, float] = None
    is_analyzed: bool = False
    is_summarized: bool = False
    thumbnail: str = None
    page_count: int = 0

    def __post_init__(self):
        if self.section_scores is None:
            self.section_scores = {}

class PitchDeckAnalyzer:
    def __init__(self):
        self.pitch_decks: Dict[str, PitchDeck] = {}
        self.sections = {
            "Problem": 0.15,
            "Solution": 0.20,
            "Market": 0.15,
            "Business Model": 0.15,
            "Financials": 0.20,
            "Team": 0.15
        }
        self.use_ocr = True
        self.ocr_resolution = 300
        
    def extract_text_from_pdf(self, pdf_file) -> Tuple[str, str, int]:
        try:
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                self.use_ocr = True
            except Exception as e:
                logger.warning(f"Could not set Tesseract path: {str(e)}")
                self.use_ocr = False
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                temp_pdf.write(pdf_file.read())
                temp_pdf_path = temp_pdf.name
            
            pdf_file.seek(0)
            
            extracted_text = ""
            thumbnail_base64 = None
            page_count = 0
            
            tesseract_available = False
            if self.use_ocr:
                try:
                    test_text = pytesseract.get_tesseract_version()
                    tesseract_available = True
                    logger.info(f"Tesseract detected, version: {test_text}")
                except Exception as e:
                    logger.warning(f"Tesseract not available: {str(e)}")
                    self.use_ocr = False
            
            try:
                doc = fitz.open(temp_pdf_path)
                page_count = len(doc)
                
                if page_count > 0:
                    first_page = doc[0]
                    pix = first_page.get_pixmap(matrix=fitz.Matrix(0.2, 0.2))
                    img_data = pix.tobytes("png")
                    thumbnail_base64 = base64.b64encode(img_data).decode()
                
                pymupdf_text = ""
                for page_num, page in enumerate(doc):
                    pymupdf_text += f"--- PAGE {page_num+1} ---\n"
                    
                    page_text = page.get_text()
                    pymupdf_text += page_text + "\n\n"
                    
                    if tesseract_available and self.use_ocr and len(page_text.strip()) < 100:
                        try:
                            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                            img = Image.open(io.BytesIO(pix.tobytes("png")))
                            ocr_text = pytesseract.image_to_string(img)
                            if len(ocr_text.strip()) > len(page_text.strip()):
                                pymupdf_text += "--- OCR TEXT ---\n" + ocr_text + "\n\n"
                        except Exception as e:
                            logger.error(f"OCR error on page {page_num+1}: {str(e)}")
                
                doc.close()
                
                if pymupdf_text.strip():
                    extracted_text = pymupdf_text
            except Exception as e:
                logger.error(f"PyMuPDF extraction error: {str(e)}")
            
            if not extracted_text.strip() or len(extracted_text.split()) < 50:
                try:
                    pdf_file.seek(0)
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    page_count = len(pdf_reader.pages)
            
                    pypdf_text = ""
                    for i, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text() or ""
                        pypdf_text += f"--- PAGE {i+1} ---\n{page_text}\n\n"
                    
                    if pypdf_text.strip():
                        extracted_text = pypdf_text if len(pypdf_text) > len(extracted_text) else extracted_text
                except Exception as e:
                    logger.error(f"PyPDF2 extraction error: {str(e)}")
            
            if (not extracted_text.strip() or len(extracted_text.split()) < 50) and tesseract_available and self.use_ocr:
                try:
                    doc = fitz.open(temp_pdf_path)
                    
                    ocr_text = ""
                    for page_num, page in enumerate(doc):
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        page_text = pytesseract.image_to_string(img)
                        ocr_text += f"--- PAGE {page_num+1} ---\n{page_text}\n\n"
                    
                    doc.close()
                    
                    if ocr_text.strip():
                        extracted_text = ocr_text
                except Exception as e:
                    logger.error(f"Full OCR error: {str(e)}")
            
            extracted_text = re.sub(r'\s+', ' ', extracted_text)
            extracted_text = re.sub(r'([a-z])- ([a-z])', r'\1\2', extracted_text)
            
            try:
                os.unlink(temp_pdf_path)
            except:
                pass
                
            return extracted_text, thumbnail_base64, page_count
            
        except Exception as e:
            st.error(f"Error extracting text: {e}")
            return "", None, 0

    def extract_company_name(self, deck_text: str) -> str:
        try:
            first_section = deck_text[:5000]
            
            company_patterns = [
                r'(?:about|introducing|presenting)\s+([A-Z][A-Za-z0-9\s]+(?:Inc|LLC|Ltd|AI|Tech|Technologies|Solutions)?)',
                r'([A-Z][A-Za-z0-9\s]+(?:Inc|LLC|Ltd|AI|Tech|Technologies|Solutions)?)\s+(?:pitch deck|presentation)',
                r'(?:welcome to|proudly presents)\s+([A-Z][A-Za-z0-9\s]+)',
                r'\b([A-Z]{2,}(?:\s+[A-Z]{2,})*)\b',
                r'([A-Z][A-Za-z0-9]+(?:\s+[A-Z]?[a-zA-Z0-9]+){0,3})\s*[:|–|-]\s*.{5,50}',
                r'(?:contact|email|website)(?:.{1,30})?([A-Z][A-Za-z0-9]+(?:\s+[A-Z]?[a-zA-Z0-9]+){0,3})(?:.{1,30})?(?:\.com|\.io|\.ai)',
            ]
            
            domains = ['.com', '.io', '.ai', '.org', '.net', '.co']
            domain_pattern = r'([a-zA-Z0-9]+(?:[a-zA-Z0-9-]+)*)(?:' + '|'.join([re.escape(d) for d in domains]) + ')'
            domain_matches = re.findall(domain_pattern, deck_text)
            potential_companies_from_domains = [m for m in domain_matches if len(m) > 2 and m.lower() not in ['www', 'http', 'https', 'email', 'mail']]
            
            pattern_matches = []
            for pattern in company_patterns:
                matches = re.findall(pattern, first_section, re.IGNORECASE)
                pattern_matches.extend([m.strip() for m in matches if len(m.strip()) > 1])
                
            pattern_matches.extend(potential_companies_from_domains)
            
            exact_formats = [
                r'^([A-Z][a-zA-Z0-9]+(?:\s[A-Z][a-zA-Z0-9]+){0,3})$',
                r'^([A-Z]{2,})$'
            ]
            
            exact_matches = []
            for pattern in exact_formats:
                for match in pattern_matches:
                    if re.match(pattern, match):
                        exact_matches.append(match)
            
            extraction_prompt = f"""
            Extract the company name from this pitch deck. Be precise and return ONLY the company name.
            
            Guidelines:
            - Company names are typically found in the title slide or "About" sections
            - They may appear in ALL CAPS or with special formatting
            - Distinguish between company names and people's names (founders, team members)
            - Common company name endings include: Inc, LLC, Ltd, AI, Technologies, Solutions
            - Look for repeated mentions of the same name throughout the text
            - Domain names may contain the company name (e.g., companyname.com)
            
            Here are potential company names found in the text (some may be incorrect):
            {', '.join(pattern_matches[:15] if pattern_matches else 'No pattern matches found')}
            
            If these are found as exact matches to common formats, consider these strongly:
            {', '.join(exact_matches[:5] if exact_matches else 'No exact format matches')}
            
            Pitch deck text excerpt:
            {first_section[:2000]}
            
            Return ONLY the company name, no explanations or additional text.
            """
            
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are an expert at identifying company names in business documents. Extract only the company name from pitch decks."},
                            {"role": "user", "content": extraction_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=20
                    )
                    company_name = response.choices[0].message.content.strip()
            
                    company_name = re.sub(r'(company name:)|(name:)|(company:)', '', company_name, flags=re.IGNORECASE).strip()
                    company_name = company_name.strip('"\'')
            
                    problematic_words = ["unclear", "unknown", "not specified", "not provided", "cannot determine"]
                    if any(word in company_name.lower() for word in problematic_words) or len(company_name) < 2:
                        cap_words = re.findall(r'\b[A-Z][a-zA-Z]*\b', deck_text)
                        cap_word_freq = {}
                
                        for word in cap_words:
                            if len(word) > 1:
                                cap_word_freq[word] = cap_word_freq.get(word, 0) + 1
                
                        sorted_words = sorted(cap_word_freq.items(), key=lambda x: x[1], reverse=True)
                
                        common_words = {"THE", "AND", "FOR", "WITH", "OUR", "WE", "THIS", "THAT", "WILL", "CAN", "MAY"}
                        filtered_words = [word for word, freq in sorted_words if word.upper() not in common_words]
                
                        if filtered_words:
                            cap_phrases = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)+\b', deck_text)
                            if cap_phrases:
                                phrase_freq = {}
                                for phrase in cap_phrases:
                                    if 2 <= len(phrase.split()) <= 4:
                                        phrase_freq[phrase] = phrase_freq.get(phrase, 0) + 1
                        
                                if phrase_freq:
                                    most_common_phrase = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)[0][0]
                                    company_name = most_common_phrase
                                else:
                                    company_name = filtered_words[0]
                            else:
                                company_name = filtered_words[0]
            
                    name_parts = company_name.split()
                    looks_like_person = (
                        len(name_parts) == 2 and 
                        all(part[0].isupper() and part[1:].islower() for part in name_parts)
                    )
            
                    if looks_like_person:
                        verification_prompt = f"""
                        Is "{company_name}" a person's name or a company name?
                        Context: This is from a pitch deck for a startup.
                        Answer with ONLY "person" or "company".
                        """
                
                        verify_response = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[{"role": "user", "content": verification_prompt}],
                            temperature=0.1,
                            max_tokens=10
                        )
                
                        if "person" in verify_response.choices[0].message.content.lower():
                            all_caps_words = re.findall(r'\b[A-Z]{2,}\b', deck_text)
                            all_caps_freq = {}
                    
                            for word in all_caps_words:
                                if len(word) > 1:
                                    all_caps_freq[word] = all_caps_freq.get(word, 0) + 1
                    
                            sorted_all_caps = sorted(all_caps_freq.items(), key=lambda x: x[1], reverse=True)
                    
                            common_words = {"THE", "AND", "FOR", "WITH", "OUR", "WE", "THIS", "THAT", "WILL", "CAN", "MAY"}
                            filtered_all_caps = [word for word, freq in sorted_all_caps if word not in common_words]
                    
                            if filtered_all_caps:
                                company_name = filtered_all_caps[0]
            
                    if len(company_name) > 30 or len(company_name.split()) > 5:
                        reasonable_phrases = re.findall(r'\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z]?[a-zA-Z0-9]*){0,3}\b', deck_text)
                        if reasonable_phrases:
                            phrase_freq = {}
                            for phrase in reasonable_phrases:
                                if 2 <= len(phrase.split()) <= 4:
                                    phrase_freq[phrase] = phrase_freq.get(phrase, 0) + 1
                    
                            if phrase_freq:
                                company_name = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)[0][0]
            
                    return company_name
                
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Company name extraction error (attempt {attempt+1}): {str(e)}. Retrying...")
                        time.sleep(retry_delay * (2 ** attempt))
                    else:
                        logger.error(f"Failed to extract company name after {max_retries} attempts: {str(e)}")
                        return f"Company_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        except Exception as e:
            st.error(f"Error extracting company name: {str(e)}")
            return f"Company_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def analyze_pitch_deck(self, deck_id: str) -> Dict[str, str]:
        try:
            deck = self.pitch_decks[deck_id]
            
            if deck.is_analyzed:
                return {"evaluation": deck.evaluation}
            
            if not deck.company_name or deck.company_name.startswith("Company_"):
                deck.company_name = self.extract_company_name(deck.deck_text)
            
            prompt = f"""
            Perform a detailed analysis of this pitch deck for {deck.company_name}. Extract ALL relevant information and provide a structured evaluation.
            
            Your output MUST follow this EXACT format:
            
            OVERALL_SCORE: [number between 0-100]
            
            SECTION_SCORES:
            Problem: [number between 0-100]
            Solution: [number between 0-100]
            Market: [number between 0-100]
            Business Model: [number between 0-100]
            Financials: [number between 0-100]
            Team: [number between 0-100]
            
            PITCH DECK STRUCTURE:
            [List the key sections present in the pitch deck and their completeness]
            
            PROBLEM STATEMENT:
            [Detailed summary of the problem being addressed]
            
            SOLUTION OVERVIEW:
            [Comprehensive description of the proposed solution]
            
            MARKET ANALYSIS:
            [Extract all market data, TAM/SAM/SOM, trends, and growth projections]
            
            BUSINESS MODEL:
            [Detail the revenue model, pricing strategy, and customer acquisition approach]
            
            FINANCIAL PROJECTIONS:
            [Extract all financial figures, projections, and metrics]
            
            TEAM BACKGROUND:
            [Summarize the team's expertise, experience, and relevant credentials]
            
            COMPETITIVE LANDSCAPE:
            [Identify competitors and competitive advantages mentioned]
            
            GO-TO-MARKET STRATEGY:
            [Summarize the planned approach to market entry and customer acquisition]
            
            TRACTION & MILESTONES:
            [Extract any metrics on current traction and key milestones]
            
            INVESTMENT ASK:
            [Detail the funding request, use of funds, and valuation if mentioned]
            
            STRENGTHS:
            - [List at least 5 key strengths with explanations]
            
            WEAKNESSES:
            - [List at least 5 areas for improvement with recommendations]
            
            INVESTMENT READINESS:
            [Assess if the pitch is ready for investment, with specific evidence]
            
            RECOMMENDATIONS:
            [Provide at least 5 specific, actionable recommendations for improvement]
            
            IMPORTANT: Extract ALL financial figures, metrics, market sizes, and quantitative data mentioned in the deck.
            Use creative analysis if the presentation is not conventional or lacks standard sections.
            """
            
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are an expert pitch deck analyst with deep experience in venture capital and startups. Your analysis should be thorough, data-driven, and comprehensive, extracting ALL relevant information from pitch decks."},
                            {"role": "user", "content": prompt},
                            {"role": "user", "content": deck.deck_text[:20000]}
                        ],
                        temperature=0.4,
                        max_tokens=4000
                    )
            
                    evaluation = response.choices[0].message.content
            
                    overall_score_match = re.search(r'OVERALL_SCORE:\s*(\d+(?:\.\d+)?)', evaluation)
                    if overall_score_match:
                        deck.pitch_score = float(overall_score_match.group(1))
                    else:
                        deck.pitch_score = 0.0
                
                    section_scores = {}
                    for section in self.sections.keys():
                        pattern = rf'{section}:\s*(\d+(?:\.\d+)?)'
                        match = re.search(pattern, evaluation)
                        if match:
                            section_scores[section] = float(match.group(1))
                        else:
                            section_scores[section] = 0.0
            
                    deck.section_scores = section_scores
                    deck.evaluation = evaluation
                    deck.is_analyzed = True
            
                    return {"evaluation": evaluation}
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Analysis error (attempt {attempt+1}): {str(e)}. Retrying...")
                        time.sleep(retry_delay * (2 ** attempt))
                    else:
                        logger.error(f"Failed to analyze pitch deck after {max_retries} attempts: {str(e)}")
                        return {"error": f"Error analyzing pitch deck: {str(e)}"}
                        
        except Exception as e:
            st.error(f"Error analyzing pitch deck: {str(e)}")
            return {"error": f"Error analyzing pitch deck: {str(e)}"}

    def get_deck_summary(self, deck_id: str) -> Dict[str, str]:
        try:
            deck = self.pitch_decks.get(deck_id)
            if not deck:
                return {"error": "Pitch deck not found"}

            if deck.is_summarized:
                return {"summary": deck.summary}

            summary_prompt = f"""
            Create a comprehensive, well-structured summary of this pitch deck, extracting ALL key information.
            Follow this EXACT format:
            
            COMPANY OVERVIEW:
            - Company Name: [Extract exact company name from the pitch deck]
            - Industry: [Specific industry sector and sub-sector]
            - Founded: [Year founded if mentioned]
            - Location: [Headquarters location if mentioned]
            - Core Value Proposition: [Concise statement of the main value proposition]
            - Stage of Development: [Precise stage of company/product development]
            
            PRODUCT/SERVICE:
            - Core Offering: [Detailed description of the main product/service]
            - Key Features: [List the main features or capabilities]
            - Technology: [Technical foundation or proprietary technology]
            - Differentiation: [What makes the product/service unique]
            
            PROBLEM & SOLUTION:
            - Problem: [Detailed description of the problem being solved]
            - Solution: [How the company's offering addresses this problem]
            - Benefits: [Specific benefits or improvements provided]
            
            MARKET OPPORTUNITY:
            - Total Addressable Market (TAM): [Extract exact market size with figures]
            - Target Market: [Specific customer segments being targeted]
            - Market Trends: [Key industry trends supporting the opportunity]
            - Growth Projections: [Market growth predictions if mentioned]
            
            BUSINESS MODEL:
            - Revenue Model: [How the company generates revenue]
            - Pricing Strategy: [Pricing information if mentioned]
            - Customer Acquisition: [Go-to-market and customer acquisition approach]
            - Sales Channels: [Distribution and sales channels]
            
            COMPETITION:
            - Primary Competitors: [List of main competitors]
            - Competitive Advantages: [Company's advantages over competitors]
            - Market Positioning: [How the company positions itself in the market]
            
            TRACTION:
            - Current Customers/Users: [Extract exact user/customer numbers if mentioned]
            - Key Metrics: [Growth rates, engagement, retention, etc.]
            - Partnerships: [Strategic partnerships or clients]
            - Milestones: [Significant achievements to date]
            
            TEAM:
            - Founders: [Background of founding team]
            - Key Personnel: [Other significant team members]
            - Advisors/Board: [Notable advisors or board members if mentioned]
            - Team Strengths: [Relevant expertise and experience]
            
            FINANCIALS:
            - Revenue: [Current or projected revenue figures]
            - Costs: [Cost structure and major expenses]
            - Margins: [Profit margins if mentioned]
            - Projections: [Key financial projections for coming years]
            - Unit Economics: [Customer acquisition cost, lifetime value, etc.]
            
            INVESTMENT ASK:
            - Funding Required: [Exact amount of funding sought]
            - Pre-money Valuation: [Valuation if mentioned]
            - Use of Funds: [Detailed breakdown of how funding will be used]
            - Previous Funding: [Prior investments if mentioned]
            - Expected Runway: [How long the funding will last]
            
            IMPORTANT INSTRUCTIONS:
            1. Extract EVERY numerical figure, metric, and data point mentioned in the deck
            2. For sections where information is not provided, indicate "Not mentioned in the deck"
            3. Include ALL relevant information without omitting key details
            4. Be precise and factual, focusing on extracting information rather than providing opinion
            5. The company name is the organization being pitched (look at title slides, headers, and logo)
            """

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting and organizing COMPREHENSIVE information from pitch decks. Your goal is to capture ALL relevant data, figures, and details."},
                    {"role": "user", "content": summary_prompt},
                    {"role": "user", "content": deck.deck_text[:12000]}
                ],
                temperature=0.3,
                max_tokens=3000
            )
            
            deck.summary = response.choices[0].message.content
            deck.is_summarized = True
            
            self.reconcile_company_name(deck_id)
            
            return {"summary": deck.summary}
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")
            return {"error": str(e)}

    def ask_question(self, deck_id: str, question: str) -> str:
        try:
            deck = self.pitch_decks.get(deck_id)
            if not deck:
                return "Error: Pitch deck not found"

            question_prompt = f"""
            Answer the following question about {deck.company_name}'s pitch deck:
            
            {question}
            
            IMPORTANT GUIDELINES:
            1. Be specific and detailed, using exact figures and data from the deck when relevant
            2. Base your answer ONLY on information present in the pitch deck
            3. If the pitch deck doesn't contain information to answer the question, clearly state this
            4. Extract ALL relevant information from the deck related to the question
            5. Include contextual information that helps provide a complete answer
            
            Pitch Deck Content:
            {deck.deck_text[:10000]}
            """

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a pitch deck analysis expert who provides detailed, comprehensive, and accurate answers based strictly on the content of pitch decks."},
                    {"role": "user", "content": question_prompt}
                ],
                temperature=0.4,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

    def reconcile_company_name(self, deck_id: str) -> None:
        deck = self.pitch_decks.get(deck_id)
        if not deck or not deck.is_summarized:
            return
        
        company_in_summary = None
        
        company_name_match = re.search(r'company name:?\s*([^,\n]+)', deck.summary, re.IGNORECASE)
        if company_name_match:
            company_in_summary = company_name_match.group(1).strip()
        
        if not company_in_summary:
            first_lines = deck.summary.split('\n')[:10]
            for line in first_lines:
                if ':' in line and ('company' in line.lower() or 'name' in line.lower()):
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        name_part = parts[1].strip()
                        if ',' in name_part:
                            name_part = name_part.split(',')[0].strip()
                        
                        if name_part and len(name_part) > 1:
                            company_in_summary = name_part
                            break
        
        if not company_in_summary or company_in_summary == deck.company_name:
            return
        
        try:
            deck_text = deck.deck_text
            
            current_name_matches = []
            summary_name_matches = []
            
            for i, match in enumerate(re.finditer(re.escape(deck.company_name), deck_text)):
                if i >= 3:
                    break
                start = max(0, match.start() - 40)
                end = min(len(deck_text), match.end() + 40)
                current_name_matches.append(f"...{deck_text[start:end]}...")
            
            for i, match in enumerate(re.finditer(re.escape(company_in_summary), deck_text)):
                if i >= 3:
                    break
                start = max(0, match.start() - 40)
                end = min(len(deck_text), match.end() + 40)
                summary_name_matches.append(f"...{deck_text[start:end]}...")
            
            decision_prompt = f"""
            Which is the correct company name for this pitch deck?
            
            Option 1: "{deck.company_name}"
            Option 2: "{company_in_summary}"
            
            Instances of Option 1 in the text:
            {current_name_matches if current_name_matches else "No direct matches found"}
            
            Instances of Option 2 in the text:
            {summary_name_matches if summary_name_matches else "No direct matches found"}
            
            Consider:
            - Which name appears in title slides or company descriptions?
            - Which name appears more consistently throughout the deck?
            - Is either name more likely to be a person's name vs. a company name?
            
            Return ONLY the number of the correct option (1 or 2).
            """
            
            decision_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": decision_prompt}],
                temperature=0.1,
                max_tokens=5
            )
            
            decision = decision_response.choices[0].message.content.strip()
            
            if "2" in decision:
                st.info(f"Updated company name from '{deck.company_name}' to '{company_in_summary}' based on validation")
                deck.company_name = company_in_summary
            
        except Exception as e:
            st.error(f"Error reconciling company name: {str(e)}")
            if company_in_summary and 2 <= len(company_in_summary.split()) <= 5:
                deck.company_name = company_in_summary

def main():
    st.set_page_config(
        page_title="AI Pitch Deck Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 AI Pitch Deck Analyzer")
    
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = PitchDeckAnalyzer()
    if 'selected_deck' not in st.session_state:
        st.session_state.selected_deck = None

    # Set Tesseract path and check availability
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    tesseract_available = False
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        version = pytesseract.get_tesseract_version()
        tesseract_available = True
        logger.info(f"Using Tesseract from: {tesseract_path}")
    except Exception as e:
        logger.warning(f"Tesseract not available at {tesseract_path}: {str(e)}")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        if not tesseract_available:
            st.warning("⚠️ Tesseract OCR is not available. Please ensure Tesseract is installed and the path is correct.\n\n" + 
                     "After installation, try using: `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`\n\n" +
                     "OCR features are currently disabled.")
            
            # Add a manual path option for tesseract
            custom_path = st.text_input("Specify correct Tesseract executable path:", 
                                      value=tesseract_path,
                                      placeholder="e.g., C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
            
            if custom_path and os.path.exists(custom_path):
                try:
                    pytesseract.pytesseract.tesseract_cmd = custom_path
                    # Test if it works
                    version = pytesseract.get_tesseract_version()
                    st.success(f"✅ Tesseract found! Version: {version}")
                    tesseract_available = True
                    use_ocr = st.checkbox("Enable OCR for image-based PDFs", value=True)
                except Exception as e:
                    st.error(f"Error with specified Tesseract path: {str(e)}")
                    use_ocr = False
            else:
                use_ocr = False
                st.checkbox("Enable OCR for image-based PDFs", value=False, disabled=True, 
                          help="Install Tesseract OCR and provide the correct path to enable this feature")
        else:
            st.success(f"✅ Tesseract OCR available (version {version})")
            use_ocr = st.checkbox("Enable OCR for image-based PDFs", value=True)
            
        if 'analyzer' in st.session_state:
            st.session_state.analyzer.use_ocr = use_ocr and tesseract_available
        
        st.header("📤 Upload Pitch Decks")
        uploaded_files = st.file_uploader("Upload PDF Pitch Decks", type="pdf", accept_multiple_files=True)
        
        if st.button("🔄 Re-analyze All Decks"):
            if st.session_state.analyzer.pitch_decks:
                for deck_id in st.session_state.analyzer.pitch_decks:
                    st.session_state.analyzer.pitch_decks[deck_id].is_analyzed = False
                    st.session_state.analyzer.pitch_decks[deck_id].is_summarized = False
                st.success("All decks are ready for reanalysis")

    if uploaded_files:
        for file in uploaded_files:
            deck_id = f"deck_{file.name}"
            
            if deck_id not in st.session_state.analyzer.pitch_decks:
                with st.spinner(f"Processing {file.name}..."):
                    deck_text, thumbnail, page_count = st.session_state.analyzer.extract_text_from_pdf(file)
                    if deck_text:
                        company_name = st.session_state.analyzer.extract_company_name(deck_text)
                        st.session_state.analyzer.pitch_decks[deck_id] = PitchDeck(
                            id=deck_id,
                            company_name=company_name,
                            deck_text=deck_text,
                            thumbnail=thumbnail,
                            page_count=page_count
                        )
                        st.success(f"Successfully processed {file.name}")
                    else:
                        st.error(f"Failed to extract text from {file.name}")

    if st.session_state.analyzer.pitch_decks:
        st.header("🔍 Pitch Deck Analysis")
        
        deck_cols = st.columns(4)
        decks_list = list(st.session_state.analyzer.pitch_decks.values())
        
        for i, deck in enumerate(decks_list):
            col_idx = i % 4
            with deck_cols[col_idx]:
                if deck.thumbnail:
                    st.image(f"data:image/png;base64,{deck.thumbnail}", caption=f"{deck.company_name} ({deck.page_count} pages)")
                else:
                    st.markdown(f"**{deck.company_name}** ({deck.page_count} pages)")
                
                if st.button(f"Select", key=f"select_{deck.id}"):
                    st.session_state.selected_deck = deck.id
        
        unanalyzed_decks = {
            did: deck 
            for did, deck in st.session_state.analyzer.pitch_decks.items() 
            if not deck.is_analyzed
        }
        
        if unanalyzed_decks:
            with st.spinner("Analyzing new pitch decks..."):
                for deck_id, deck in unanalyzed_decks.items():
                    analysis = st.session_state.analyzer.analyze_pitch_deck(deck_id)
                    if "error" not in analysis:
                        st.success(f"Analyzed {deck.company_name}'s pitch deck")

        st.subheader("📈 Analysis Results")
        analyzed_decks = sorted(
            [d for d in st.session_state.analyzer.pitch_decks.values() if d.is_analyzed],
            key=lambda x: x.pitch_score,
            reverse=True
        )

        if analyzed_decks:
            for rank, deck in enumerate(analyzed_decks, 1):
                with st.expander(f"#{rank} - {deck.company_name} (Overall Score: {deck.pitch_score:.1f}/100)"):
                    cols = st.columns(6)
                    for i, (section, score) in enumerate(deck.section_scores.items()):
                        cols[i % 6].metric(section, f"{score:.1f}")
                    
                    if deck.thumbnail:
                        st.image(f"data:image/png;base64,{deck.thumbnail}", width=200)
                    
                    st.divider()
                    if st.button(f"View Detailed Analysis - {deck.company_name}", key=f"view_{deck.id}"):
                        st.session_state.selected_deck = deck.id

        if st.session_state.selected_deck:
            deck = st.session_state.analyzer.pitch_decks[st.session_state.selected_deck]
            
            st.markdown("---")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.header(f"{deck.company_name} - Detailed Analysis")
            with col2:
                if deck.is_analyzed:
                    st.metric("Overall Score", f"{deck.pitch_score:.1f}/100")
                else:
                    st.info("This deck has not been analyzed yet.")
            
            if deck.thumbnail:
                st.image(f"data:image/png;base64,{deck.thumbnail}", width=300)
            
            st.divider()
            
            if not deck.is_summarized:
                with st.spinner("Generating summary..."):
                    summary = st.session_state.analyzer.get_deck_summary(deck.id)
                    st.write(summary.get("summary", "Error getting summary"))
            else:
                st.write(deck.summary)
            
            st.divider()
            
            if deck.is_analyzed and deck.section_scores is not None:
                st.subheader("Section Scores")
                score_cols = st.columns(6)
                for i, (section, score) in enumerate(deck.section_scores.items()):
                    color = "green" if score >= 70 else "orange" if score >= 50 else "red"
                    score_cols[i % 6].markdown(f"**{section}**: <span style='color:{color}'>{score:.1f}</span>", unsafe_allow_html=True)
            
            if deck.is_analyzed:
                with st.expander("View Complete Analysis"):
                    st.write(deck.evaluation)
            else:
                st.warning("Section scores unavailable. The analysis might be incomplete.")
                with st.expander("View Available Analysis"):
                    st.write(deck.evaluation or "No analysis available.")
            
            st.subheader("💬 Ask Questions")
            question = st.text_input("Enter your question about this pitch deck:")
            if st.button("Ask Question"):
                if question:
                    with st.spinner("Getting answer..."):
                        answer = st.session_state.analyzer.ask_question(deck.id, question)
                        st.markdown("### Answer:")
                        st.write(answer)

if __name__ == "__main__":
    main()