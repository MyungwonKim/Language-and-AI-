import pandas as pd
import re
import html
import ftfy
from langdetect import detect, LangDetectException
from tqdm import tqdm 

INPUT_FILE = "data/extrovert_introvert.csv"
OUTPUT_FILE = "data/processed_reddit_posts.csv"
AUTHOR_COL = "auhtor_ID"   
POST_COL = "post"       
LABEL_COL = "extrovert" 

# Class for preprocessing 
class RedditPreprocessor:
    def __init__(self):
        self.url_pattern = re.compile(r'http\S+|www\.\S+')
        self.user_pattern = re.compile(r'u/\S+')
        self.sub_pattern = re.compile(r'r/\S+')
        
        # Regex for quotes
        self.quote_pattern = re.compile(r'^\s*>.*$', re.MULTILINE)
        
        self.symbol_squash_pattern = re.compile(r'([!?.@$])\1{2,}')
        self.markdown_link_pattern = re.compile(r'\[(.*?)\]\(.*?\)')
        
        self.bot_phrases = [
            "i am a bot", "action was performed automatically", 
            "submission has been removed", "contact the moderators"
        ]

    def clean_post(self, text):
        if not isinstance(text, str): return ""
        
        # Fix encoding & HTML
        text = ftfy.fix_text(text)
        text = html.unescape(text)
        
        # Remove quotes & markdown links
        text = self.quote_pattern.sub('', text)
        text = self.markdown_link_pattern.sub(r'\1', text)
        
        # Bot check
        if any(phrase in text.lower() for phrase in self.bot_phrases):
            return ""

        # Token replacements
        text = self.url_pattern.sub('[URL]', text)
        text = self.user_pattern.sub('[USER]', text)
        text = self.sub_pattern.sub('[SUB]', text)
        
        # Symbol squashing & whitespace
        text = self.symbol_squash_pattern.sub(r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Language check
        if len(text) < 5: 
            return text if text.isascii() else ""
            
        try:
            if detect(text) != 'en': return ""
        except LangDetectException:
            return ""

        return text

# Main execution
def process_data():
    print("Loading data...")
    try:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_FILE, encoding='latin1')
        
    print(f"Original shape: {df.shape}")
    
    # Initialize preprocessor
    processor = RedditPreprocessor()
    tqdm.pandas(desc="Cleaning Posts")
    
    # Apply cleaning
    print("Cleaning posts...")
    df['clean_text'] = df[POST_COL].progress_apply(processor.clean_post)
    
    # Remove empty rows
    # This removes posts that were filtered out by language check or bot check
    initial_count = len(df)
    df = df[df['clean_text'] != ""]
    print(f"Removed {initial_count - len(df)} empty/non-English rows.")
    
    # Final selection
    cols_to_keep = [AUTHOR_COL, 'clean_text']
    
    if LABEL_COL in df.columns:
        cols_to_keep.append(LABEL_COL)
    
    df_final = df[cols_to_keep]
    
    print(f"Final shape: {df_final.shape}")
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved processed data to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_data()
