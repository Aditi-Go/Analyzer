import json
import codecs
from google import genai
from typing import List, Dict, Any
import pandas as pd
import streamlit as st
from pathlib import Path
import asyncio
from functools import lru_cache
import os
from dotenv import load_dotenv
import time
import random

load_dotenv()

class InvestorMatchingSystem:
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash"
    
    @staticmethod
    def load_data(file_path: str) -> Dict:
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252']
            data = None
            
            for encoding in encodings:
                try:
                    with codecs.open(file_path, 'r', encoding=encoding) as file:
                        data = json.load(file)
                    break
                except UnicodeDecodeError:
                    continue
                except json.JSONDecodeError:
                    continue
            
            if data is None:
                st.error("Could not decode the JSON file with any encoding.")
                return {'founders': [], 'investors': []}
            
            founders = []
            investors = []
            
            for item in data:
                normalized_item = {k.lower(): v for k, v in item.items()}
                
                if 'name' in normalized_item and 'id' not in normalized_item:
                    normalized_item['id'] = normalized_item['name'].replace(' ', '_')
                
                if normalized_item.get('type', '').lower() in ['startup', 'founder', 'company']:
                    founders.append({
                        'id': normalized_item.get('id', f"F{len(founders)+1}"),
                        'name': normalized_item.get('name', 'Unknown'),
                        'industry': normalized_item.get('industry', ''),
                        'stage': normalized_item.get('stage', ''),
                        'funding_required': normalized_item.get('cheque_range', normalized_item.get('funding_required', '')),
                        'traction': normalized_item.get('overview', ''),
                        'business_model': normalized_item.get('industry', '')
                    })
                else:
                    investors.append({
                        'id': normalized_item.get('id', f"I{len(investors)+1}"),
                        'name': normalized_item.get('name', 'Unknown'),
                        'preferred_industry': normalized_item.get('industry', ''),
                        'investment_range': normalized_item.get('cheque_range', ''),
                        'stage_preference': normalized_item.get('stage', '')
                    })
            
            if not founders and not investors:
                for item in data:
                    normalized_item = {k.lower(): v for k, v in item.items()}
                    
                    if 'name' in normalized_item and 'id' not in normalized_item:
                        normalized_item['id'] = normalized_item['name'].replace(' ', '_')
                    
                    if ('cheque_range' in normalized_item or 
                        'investment_range' in normalized_item or 
                        normalized_item.get('type', '').lower() in ['vc', 'family office', 'investor']):
                        investors.append({
                            'id': normalized_item.get('id', f"I{len(investors)+1}"),
                            'name': normalized_item.get('name', 'Unknown'),
                            'preferred_industry': normalized_item.get('industry', ''),
                            'investment_range': normalized_item.get('cheque_range', ''),
                            'stage_preference': normalized_item.get('stage', '')
                        })
                    else:
                        founders.append({
                            'id': normalized_item.get('id', f"F{len(founders)+1}"),
                            'name': normalized_item.get('name', 'Unknown'),
                            'industry': normalized_item.get('industry', ''),
                            'stage': normalized_item.get('stage', ''),
                            'funding_required': normalized_item.get('cheque_range', ''),
                            'traction': normalized_item.get('overview', ''),
                            'business_model': normalized_item.get('industry', '')
                        })
            
            if not founders and investors:
                founders.append({
                    'id': 'F1',
                    'name': 'Rakesh Kumar',
                    'industry': 'Technology',
                    'stage': 'Seed',
                    'funding_required': '$500K - $2M',
                    'traction': 'Early traction',
                    'business_model': 'B2B SaaS'
                })
                
                founders.extend([
                    {
                        'id': 'F2',
                        'name': 'Rajesh Sharma',
                        'industry': 'Fintech',
                        'stage': 'Series A',
                        'funding_required': '$2M - $5M',
                        'traction': 'Growing user base with 50,000 monthly active users',
                        'business_model': 'B2C Financial Services'
                    },
                    {
                        'id': 'F3',
                        'name': 'Priya Patel',
                        'industry': 'Healthcare',
                        'stage': 'Seed',
                        'funding_required': '$500K - $1M',
                        'traction': 'Pilot with 3 major hospitals in Bangalore',
                        'business_model': 'B2B Healthcare SaaS'
                    },
                    {
                        'id': 'F4',
                        'name': 'Vikram Mehta',
                        'industry': 'E-commerce',
                        'stage': 'Pre-seed',
                        'funding_required': '$100K - $500K',
                        'traction': 'MVP with 1,000 early adopters',
                        'business_model': 'D2C E-commerce'
                    },
                    {
                        'id': 'F5',
                        'name': 'Ananya Desai',
                        'industry': 'EdTech',
                        'stage': 'Series B',
                        'funding_required': '$10M - $20M',
                        'traction': '500,000 students across 200 schools in India',
                        'business_model': 'B2B2C Education Platform'
                    }
                ])
            
            if not investors and founders:
                investors.append({
                    'id': 'I1',
                    'name': 'Sample Investor',
                    'preferred_industry': 'Technology',
                    'investment_range': '$250K - $2M',
                    'stage_preference': 'Seed'
                })
            
            return {
                'founders': founders,
                'investors': investors
            }
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return {'founders': [], 'investors': []}

    async def calculate_match_score(self, founder: Dict, investor: Dict) -> float:
        """Calculate match score using Gemini API with retry mechanism"""
        prompt = f"""
        Analyze the compatibility between this founder and investor:
        
        Founder:
        - Industry: {founder.get('industry', 'N/A')}
        - Stage: {founder.get('stage', 'N/A')}
        - Funding Required: {founder.get('funding_required', 'N/A')}
        - Traction: {founder.get('traction', 'N/A')}
        - Business Model: {founder.get('business_model', 'N/A')}
        
        Investor:
        - Preferred Industry: {investor.get('preferred_industry', 'N/A')}
        - Investment Range: {investor.get('investment_range', 'N/A')}
        - Stage Preference: {investor.get('stage_preference', 'N/A')}
        
        Provide a compatibility score between 0 and 1, where 1 is a perfect match.
        Only respond with the numerical score.
        """
        
        # Define retry parameters
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )
                
                # Try to convert response to float
                try:
                    score = float(response.text.strip())
                    return min(max(score, 0), 1)  # Ensure score is between 0 and 1
                except (ValueError, AttributeError) as e:
                    st.warning(f"Failed to parse score: {response.text}. Using fallback calculation.")
                    
                    # Fallback: Calculate a basic score based on direct comparison
                    score = self._calculate_fallback_score(founder, investor)
                    return score
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = retry_delay * (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                    st.warning(f"API error (attempt {attempt+1}/{max_retries}): {str(e)}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    st.warning(f"Failed to get API response after {max_retries} attempts. Using fallback calculation.")
                    
                    # Fallback: Calculate a basic score without using the API
                    score = self._calculate_fallback_score(founder, investor)
                    return score
        
        # If we somehow get here, use fallback
        return self._calculate_fallback_score(founder, investor)
    
    def _calculate_fallback_score(self, founder: Dict, investor: Dict) -> float:
        """Calculate a basic match score without using the API"""
        score = 0.5  # Default middle score
        
        # 1. Industry match
        founder_industry = founder.get('industry', '').lower()
        investor_industry = investor.get('preferred_industry', '').lower()
        
        if founder_industry and investor_industry:
            # Check if there's overlap in industries
            if founder_industry in investor_industry or investor_industry in founder_industry:
                score += 0.2
            elif any(ind in founder_industry for ind in investor_industry.split(',')):
                score += 0.15
            else:
                score -= 0.1
        
        # 2. Stage match
        founder_stage = founder.get('stage', '').lower()
        investor_stage = investor.get('stage_preference', '').lower()
        
        if founder_stage and investor_stage:
            if founder_stage in investor_stage or investor_stage in founder_stage:
                score += 0.2
            else:
                score -= 0.1
        
        # 3. Funding range match
        founder_funding = founder.get('funding_required', '').lower()
        investor_range = investor.get('investment_range', '').lower()
        
        # Simplistic check for overlap in funding ranges
        if founder_funding and investor_range:
            if any(amount in investor_range for amount in founder_funding.split('-')):
                score += 0.1
        
        # Ensure score is between 0 and 1
        return min(max(score, 0), 1)

    async def find_matches(self, founder_id: str, data: Dict) -> List[Dict]:
        """Find and rank matching investors for a given founder"""
        founder = next((f for f in data['founders'] if f.get('id') == founder_id), None)
        if not founder:
            st.error("Founder not found")
            return []

        matches = []
        progress_bar = st.progress(0)
        total_investors = len(data['investors'])
        
        # Create a status placeholder to display current investor being processed
        status_text = st.empty()

        for idx, investor in enumerate(data['investors']):
            investor_name = investor.get('name', 'Unknown Investor')
            status_text.text(f"Processing match with: {investor_name} ({idx+1}/{total_investors})")
            
            try:
                score = await self.calculate_match_score(founder, investor)
                matches.append({
                    'investor_id': investor.get('id', 'Unknown'),
                    'investor_name': investor_name,
                    'match_score': score,
                    'preferred_industry': investor.get('preferred_industry', 'Not specified'),
                    'investment_range': investor.get('investment_range', 'Not specified'),
                    'stage_preference': investor.get('stage_preference', 'Not specified')
                })
            except Exception as e:
                st.warning(f"Error processing {investor_name}: {str(e)}")
                # Still add the investor but with a zero score
                matches.append({
                    'investor_id': investor.get('id', 'Unknown'),
                    'investor_name': investor_name,
                    'match_score': 0.0,
                    'preferred_industry': investor.get('preferred_industry', 'Not specified'),
                    'investment_range': investor.get('investment_range', 'Not specified'),
                    'stage_preference': investor.get('stage_preference', 'Not specified')
                })
            
            # Update progress
            progress_bar.progress((idx + 1) / total_investors)
            
            # Add a small delay to avoid rate limits
            await asyncio.sleep(0.5)

        # Clear the status text when done
        status_text.empty()
        
        return sorted(matches, key=lambda x: x['match_score'], reverse=True)

def create_streamlit_app():
    st.set_page_config(
        page_title="Founder-Investor Matching System",
        page_icon="🤝",
        layout="wide"
    )

    st.title("🤝 Founder-Investor Matching System")

    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = os.getenv("GEMINI_API_KEY")
        
        uploaded_file = st.file_uploader("Upload Dataset (JSON)", type=['json'])
        
        if uploaded_file:
            try:
                content = uploaded_file.read()
                with open("temp_data.json", "wb") as f:
                    f.write(content)
                file_path = "temp_data.json"
            except Exception as e:
                st.error(f"Error saving file: {str(e)}")
                return
        else:
            use_sample_data = st.checkbox("Use sample data")
            
            if use_sample_data:
                sample_data = [
                    {
                        "Name": "01 Ventures",
                        "Website": "https://www.01ventures.com/",
                        "Global_HQ": "Netherlands",
                        "Stage": "Pre-seed, Idea, Prototype/MVP, Seed",
                        "Overview": "We invest in deep tech innovations including software and hardware solutions.",
                        "Type": "VC",
                        "Industry": "Information Technology & Services",
                        "Cheque_range": "$250K - $2M"
                    },
                    {
                        "Name": "Tech Startup",
                        "Website": "https://techstartup.com/",
                        "Global_HQ": "United States",
                        "Stage": "Seed",
                        "Overview": "Building innovative SaaS solutions.",
                        "Type": "Startup",
                        "Industry": "Software",
                        "Cheque_range": "$500K - $1M"
                    },
                    {
                        "Name": "Health Capital",
                        "Website": "https://healthcapital.com/",
                        "Global_HQ": "United Kingdom",
                        "Stage": "Series A, Series B",
                        "Overview": "We invest in healthcare and biotech startups.",
                        "Type": "VC",
                        "Industry": "Healthcare, Biotech",
                        "Cheque_range": "$1M - $5M"
                    }
                ]
                
                with open("sample_data.json", "w", encoding="utf-8") as f:
                    json.dump(sample_data, f, ensure_ascii=False)
                file_path = "sample_data.json"
            else:
                st.info("Please upload your dataset or use sample data")
                return

    matching_system = InvestorMatchingSystem(api_key)
    
    with st.spinner("Loading and processing data..."):
        data = matching_system.load_data(file_path)

    if not data['founders'] or not data['investors']:
        st.error("Could not identify founders and investors in your data.")
        st.info("""
        Make sure your JSON file contains records that can be identified as founders and investors.
        
        For investors, the system looks for:
        - 'Type' field containing 'VC', 'Family office', or 'Investor'
        - 'Cheque_range' or 'Investment_range' fields
        
        For founders, the system looks for:
        - 'Type' field containing 'Startup', 'Founder', or 'Company'
        - Or will assign any remaining records as founders
        """)
        return

    col_viz1, col_viz2 = st.columns(2)
    with col_viz1:
        st.subheader("Founders/Startups in Dataset")
        st.write(f"Total founders: {len(data['founders'])}")
        st.dataframe(pd.DataFrame(data['founders']).fillna(''), hide_index=True)
    
    with col_viz2:
        st.subheader("Investors in Dataset")
        st.write(f"Total investors: {len(data['investors'])}")
        st.dataframe(pd.DataFrame(data['investors']).fillna(''), hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("👨‍💼 Select Founder/Startup")
        founder_options = {f.get('name', f.get('id', 'Unknown')): f.get('id') for f in data['founders']}
        if founder_options:
            selected_founder = st.selectbox(
                "Choose a founder to match",
                options=list(founder_options.keys())
            )

            if st.button("🔍 Find Matches", type="primary"):
                founder_id = founder_options[selected_founder]
                
                with st.spinner("🔄 Finding the best investor matches..."):
                    matches = asyncio.run(matching_system.find_matches(founder_id, data))

                with col2:
                    st.subheader("🎯 Matched Investors")
                    
                    if matches:
                        df = pd.DataFrame(matches)
                        
                        def style_match_score(val):
                            if val >= 0.7:
                                return 'background-color: #90EE90'
                            elif val >= 0.4:
                                return 'background-color: #FFD700'
                            return 'background-color: #FFB6C1'

                        styled_df = df.style.applymap(
                            style_match_score, 
                            subset=['match_score']
                        )

                        st.dataframe(
                            styled_df,
                            column_config={
                                "match_score": st.column_config.ProgressColumn(
                                    "Match Score",
                                    min_value=0,
                                    max_value=1,
                                    format="%.2f"
                                )
                            },
                            hide_index=True
                        )

                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="investor_matches.csv",
                            mime="text/csv",
                        )
                        
                        st.subheader("📊 Match Statistics")
                        col_stats1, col_stats2, col_stats3 = st.columns(3)
                        with col_stats1:
                            st.metric("High Matches (>0.7)", 
                                    len(df[df['match_score'] > 0.7]))
                        with col_stats2:
                            st.metric("Medium Matches (0.4-0.7)", 
                                    len(df[(df['match_score'] >= 0.4) & (df['match_score'] <= 0.7)]))
                        with col_stats3:
                            st.metric("Low Matches (<0.4)", 
                                    len(df[df['match_score'] < 0.4]))
                    else:
                        st.info("No matches found")
        else:
            st.error("No founders found in the dataset")

if __name__ == "__main__":
    create_streamlit_app() 