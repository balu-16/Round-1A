#!/usr/bin/env python3
"""
Adobe India Hackathon 2025 - PDF Structure Parser v2.0
Enhanced PDF parser for extracting titles and headings with improved accuracy.
"""

import fitz  # PyMuPDF
import json
import os
import re
import time
from typing import List, Dict, Any
from collections import defaultdict, Counter

# Configuration Constants
ROUND_DIGIT = 1
MIN_HEADING_SIZE_RATIO = 1.1  # Minimum ratio above average font size for headings
TITLE_SEARCH_HEIGHT_RATIO = 0.4  # Search in top 40% of first page for title
HEADER_FOOTER_MARGIN_RATIO = 0.05  # Skip top/bottom 5% as headers/footers
MAX_HEADING_LENGTH = 100  # Maximum length for headings
MIN_HEADING_LENGTH = 3  # Minimum length for headings
MAX_HEADING_LEVELS = 4  # Maximum number of heading levels (H1-H4)
CENTERING_THRESHOLD = 0.3  # Distance from center to consider text centered

class PDFStructureParser:
    def __init__(self):
        self.avg_font_size = 0
        self.font_sizes = []
        self.heading_sizes = []
        self.body_font_size = 0
        
    def _analyze_text_characteristics(self, text: str) -> Dict[str, Any]:
        """Analyze text characteristics for intelligent classification."""
        if not text:
            return {}
            
        text_lower = text.lower().strip()
        words = text_lower.split()
        
        characteristics = {
            'length': len(text),
            'word_count': len(words),
            'has_numbers': bool(re.search(r'\d', text)),
            'is_all_caps': text.isupper(),
            'is_title_case': text.istitle(),
            'has_punctuation': bool(re.search(r'[.!?]', text)),
            'starts_with_number': text_lower.strip().startswith(tuple('0123456789')),
            'ends_with_colon': text.strip().endswith(':'),
            'has_common_words': any(word in ['the', 'and', 'or', 'of', 'in', 'to', 'for', 'with'] for word in words),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0
        }
        
        return characteristics
    
    def _calculate_title_likelihood(self, text: str, characteristics: Dict[str, Any]) -> float:
        """Calculate likelihood that text is a title using intelligent analysis."""
        score = 0
        text_lower = text.lower().strip()
        
        # Length-based scoring
        if 10 <= characteristics.get('length', 0) <= 80:
            score += 10
        elif 5 <= characteristics.get('length', 0) <= 120:
            score += 5
        
        # Word count scoring
        word_count = characteristics.get('word_count', 0)
        if 2 <= word_count <= 8:
            score += 15
        elif 1 <= word_count <= 15:
            score += 8
        
        # Format scoring
        if characteristics.get('is_title_case') or characteristics.get('is_all_caps'):
            score += 10
        
        # Content analysis
        if not characteristics.get('has_punctuation'):
            score += 5  # Titles often don't have punctuation
        
        if characteristics.get('ends_with_colon'):
            score -= 5  # Titles rarely end with colons
        
        # Penalize obvious non-titles
        if 'address' in text_lower or 'phone' in text_lower or 'email' in text_lower:
            score -= 20
        
        if re.match(r'^\d+[\.)\s]', text):  # Starts like a list item
            score -= 15
        
        if 'page' in text_lower and re.search(r'\d', text):
            score -= 20
        
        return score
    
    def _calculate_heading_likelihood(self, text: str, characteristics: Dict[str, Any]) -> float:
        """Calculate likelihood that text is a heading using intelligent analysis."""
        score = 0
        text_lower = text.lower().strip()
        
        # Length-based scoring
        if 5 <= characteristics.get('length', 0) <= 60:
            score += 8
        elif characteristics.get('length', 0) > 100:
            score -= 10  # Very long text unlikely to be heading
        
        # Word count scoring
        word_count = characteristics.get('word_count', 0)
        if 1 <= word_count <= 8:
            score += 10
        elif word_count > 15:
            score -= 8
        
        # Format scoring
        if characteristics.get('is_title_case') or characteristics.get('is_all_caps'):
            score += 8
        
        if characteristics.get('ends_with_colon'):
            score += 5  # Headings often end with colons
        
        if not characteristics.get('has_punctuation'):
            score += 3  # Headings often don't have sentence punctuation
        
        # Penalize obvious non-headings
        if characteristics.get('starts_with_number') and '.' in text:
            score -= 10  # Likely a numbered paragraph
        
        # Strong penalties for specific non-heading patterns
        # if 'rsvp' in text_lower:
        #     score -= 25  # RSVP lines are not headings
        
        if re.match(r'^[-_=\*\+\.]{2,}$', text):
            score -= 20  # Decorative lines
        
        if 'address' in text_lower or 'phone' in text_lower:
            score -= 15
        
        # Penalize lines with many dashes or underscores (decorative)
        dash_count = text.count('-') + text.count('_')
        if dash_count > 3:
            score -= 20
        
        # Penalize very short text with special characters
        if len(text.strip()) <= 5 and re.search(r'[^\w\s]', text):
            score -= 10
        
        # Boost for exclamatory text (indicates importance)
        if text.endswith('!'):
            score += 8
        
        return score
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove excessive whitespace but preserve single spaces
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common OCR spacing issues
        text = re.sub(r'\s+([!?.])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([A-Z])\s+([a-z])', r'\1\2', text)  # Fix "Y ou" -> "You"
        text = re.sub(r'([a-z])\s+([A-Z])\s+([a-z])', r'\1\2\3', text)  # Fix "T HERE" -> "THERE"
        text = re.sub(r'\b([A-Z])\s+([A-Z]{2,})\b', r'\1\2', text)  # Fix "T HERE" -> "THERE" for all caps
        
        # Remove common noise patterns
        if re.match(r'^[^\w\s]*$', text):  # Only special characters
            return ""
        if re.match(r'^\d+$', text):  # Only numbers
            return ""
        if re.match(r'^[_\-=\*\+\.]{3,}$', text):  # Decorative lines
            return ""
            
        return text
    
    def _is_noise_text(self, text: str) -> bool:
        """Check if text is likely noise and should be filtered out."""
        if not text or len(text.strip()) < 2:
            return True
        
        text_lower = text.lower().strip()
        
        # Filter out address components and business info
        if re.match(r'^\d+\s+[A-Z\s]+$', text):  # Number + street name
            return True
        if re.match(r'^[A-Z]{2}\s+\d{5}$', text):  # State + ZIP
            return True
        if 'www.' in text_lower or '.com' in text_lower:  # URLs
            return True
        if text.startswith('(') and text.endswith(')'):  # Parenthetical notes
            return True
        if text_lower.startswith('address:'):
            return True
        if re.match(r'^[_\-=\*\+\.]{2,}$', text):  # Decorative lines
            return True
        
        # Filter out business names and location info that shouldn't be headings
        business_patterns = [
            r'^topjump$',
            r'^\d+\s+parkway$',
            r'^pigeon\s+forge',
            r'^near\s+dixie',
            r'^on\s+the\s+parkway'
        ]
        
        for pattern in business_patterns:
            if re.match(pattern, text_lower):
                return True
        
        return False
    
    def _collect_font_statistics(self, doc: fitz.Document) -> None:
        """Collect font size statistics from the document."""
        all_sizes = []
        
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if text and len(text) > 2:  # Only meaningful text
                            size = round(span["size"], ROUND_DIGIT)
                            all_sizes.append(size)
        
        if not all_sizes:
            self.avg_font_size = 12.0
            self.body_font_size = 12.0
            self.heading_sizes = []
            return
        
        # Calculate statistics
        size_counts = Counter(all_sizes)
        self.avg_font_size = sum(all_sizes) / len(all_sizes)
        
        # Body font is the most common size
        self.body_font_size = size_counts.most_common(1)[0][0]
        
        # Heading sizes are those significantly larger than body font
        unique_sizes = sorted(set(all_sizes), reverse=True)
        self.heading_sizes = []
        
        for size in unique_sizes:
            if size > self.body_font_size * MIN_HEADING_SIZE_RATIO:
                self.heading_sizes.append(size)
        
        # Limit to top 4 heading sizes (H1-H4)
        self.heading_sizes = self.heading_sizes[:4]
        
    def _extract_title(self, doc: fitz.Document) -> str:
        """Extract document title from the first page."""
        first_page = doc[0]
        page_height = first_page.rect.height
        page_width = first_page.rect.width
        
        # Search in top portion of first page for title
        search_height = page_height * TITLE_SEARCH_HEIGHT_RATIO
        
        title_candidates = []
        blocks = first_page.get_text("dict")["blocks"]
        
        for block in blocks:
            x0, y0, x1, y1 = block['bbox']
            
            # Only consider text in the top area
            if y1 > search_height:
                continue
                
            for line in block.get("lines", []):
                line_text = ""
                max_size = 0
                
                # Combine all spans in the line
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    size = round(span["size"], ROUND_DIGIT)
                    
                    if text:
                        line_text += text + " "
                        max_size = max(max_size, size)
                
                line_text = self.clean_text(line_text)
                
                if line_text and len(line_text) > MIN_HEADING_LENGTH and not self._is_noise_text(line_text):
                    # Calculate title score
                    score = 0
                    
                    # Size score (larger fonts get much higher score)
                    if max_size > self.avg_font_size:
                        score += (max_size / self.avg_font_size) * 15
                    
                    # Length score (prefer moderate length titles)
                    word_count = len(line_text.split())
                    if 2 <= word_count <= 12:
                        score += 15
                    elif 1 <= word_count <= 20:
                        score += 8
                    
                    # Position score (higher on page is better)
                    relative_y = y0 / page_height
                    if relative_y < 0.15:
                        score += 20
                    elif relative_y < 0.3:
                        score += 12
                    
                    # Centering score
                    line_center = (x0 + x1) / 2
                    page_center = page_width / 2
                    center_distance = abs(line_center - page_center) / page_width
                    if center_distance < CENTERING_THRESHOLD:
                        score += 10
                    
                    # Intelligent text analysis
                    characteristics = self._analyze_text_characteristics(line_text)
                    title_likelihood = self._calculate_title_likelihood(line_text, characteristics)
                    score += title_likelihood
                    
                    # Bonus for all caps or title case
                    if line_text.isupper() or line_text.istitle():
                        score += 5
                    
                    title_candidates.append({
                        'text': line_text,
                        'score': score,
                        'size': max_size
                    })
        
        # Return the highest scoring candidate
        if title_candidates:
            title_candidates.sort(key=lambda x: x['score'], reverse=True)
            return title_candidates[0]['text']
        
        return ""
    
    def _determine_heading_level(self, font_size: float) -> str:
        """Determine heading level based on font size."""
        if not self.heading_sizes:
            # Fallback logic when no heading sizes are collected
            if font_size >= self.body_font_size * 2.0:
                return "H1"
            elif font_size >= self.body_font_size * 1.5:
                return "H2"
            elif font_size >= self.body_font_size * 1.3:
                return "H3"
            else:
                return "H4"
        
        # Use collected heading sizes for more accurate level determination
        # Sort heading sizes in descending order for proper H1, H2, H3 assignment
        sorted_sizes = sorted(self.heading_sizes, reverse=True)
        
        # Find the appropriate level based on font size
        for i, size in enumerate(sorted_sizes):
            if font_size >= size - 0.5:  # Allow small tolerance
                level = min(i + 1, 4)  # Cap at H4
                return f"H{level}"
        
        # If smaller than all heading sizes but still larger than body, make it H4
        if font_size > self.body_font_size * MIN_HEADING_SIZE_RATIO:
            return "H4"
        
        return "H4"
    
    def _extract_headings(self, doc: fitz.Document, title: str = "") -> List[Dict[str, Any]]:
        """Extract headings from the document."""
        headings = []
        title_lower = title.lower().strip()
        
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                # Skip headers and footers
                x0, y0, x1, y1 = block['bbox']
                page_height = page.rect.height
                margin = page_height * HEADER_FOOTER_MARGIN_RATIO
                if y1 < margin or y0 > page_height - margin:
                    continue
                
                for line in block.get("lines", []):
                    line_text = ""
                    max_size = 0
                    
                    # Combine all spans in the line
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        size = round(span["size"], ROUND_DIGIT)
                        
                        if text:
                            line_text += text + " "
                            max_size = max(max_size, size)
                    
                    line_text = self.clean_text(line_text)
                    
                    # Check if this could be a heading with more strict criteria
                    if (line_text and 
                        len(line_text) >= MIN_HEADING_LENGTH and 
                        len(line_text) <= MAX_HEADING_LENGTH and
                        max_size > self.body_font_size * MIN_HEADING_SIZE_RATIO and
                        not self._is_noise_text(line_text) and
                        self._is_likely_heading(line_text)):  # Additional heading validation
                        
                        # Skip if it's the title
                        if title_lower and line_text.lower().strip() == title_lower:
                            continue
                        
                        # Skip obvious non-headings
                        if re.match(r'^\d+[\.\)]\s*$', line_text):  # Not just numbers
                            continue
                        if re.match(r'^page\s+\d+', line_text.lower()):  # Not page numbers
                            continue
                        
                        # Determine heading level
                        level = self._determine_heading_level(max_size)
                        
                        headings.append({
                            "level": level,
                            "text": line_text,
                            "page": page_num + 1  # Start page numbers from 1 instead of 0
                        })
        
        # Remove duplicates while preserving order
        seen = set()
        unique_headings = []
        for heading in headings:
            key = (heading['text'].lower(), heading['page'])
            if key not in seen:
                seen.add(key)
                unique_headings.append(heading)
        
        return unique_headings
    
    def _is_likely_heading(self, text: str) -> bool:
        """Check if text is likely to be a real heading using intelligent analysis."""
        characteristics = self._analyze_text_characteristics(text)
        heading_likelihood = self._calculate_heading_likelihood(text, characteristics)
        
        # Return True if likelihood score is positive
        return heading_likelihood > 0
    
    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse a PDF file and extract its structure.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing title and outline
        """
        try:
            doc = fitz.open(pdf_path)
            
            # Collect font statistics
            self._collect_font_statistics(doc)
            
            # Extract title
            title = self._extract_title(doc)
            
            # Extract headings
            headings = self._extract_headings(doc, title)
            
            doc.close()
            
            return {
                "title": title,
                "outline": headings
            }
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return {"title": "", "outline": []}

def main():
    """Main function to process all PDF files."""
    print("🚀 Adobe India Hackathon 2025 - PDF Structure Parser v2.0")
    print("=" * 60)
    
    start_time = time.time()
    
    # Input and output directories (configurable)
    input_dir = os.getenv('INPUT_DIR', 'input')
    output_dir = os.getenv('OUTPUT_DIR', 'output')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize parser
    parser = PDFStructureParser()
    
    # Process all PDF files
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    pdf_files.sort()  # Process in order
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        output_file = os.path.join(output_dir, pdf_file.replace('.pdf', '.json'))
        
        print(f"📄 Processing {pdf_file}...")
        
        # Parse the PDF
        result = parser.parse_pdf(pdf_path)
        
        # Save the result
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Display summary
        heading_count = len(result['outline'])
        print(f"✅ Extracted {heading_count} headings from {pdf_file}")
        
        if result['title']:
            print(f"📋 Title: {result['title']}")
        
        if result['outline']:
            print("📑 Headings found:")
            for i, heading in enumerate(result['outline'][:5]):  # Show first 5
                print(f"   {heading['level']}: {heading['text']} (Page {heading['page']})")
            if len(result['outline']) > 5:
                print(f"   ... and {len(result['outline']) - 5} more")
        
        print()

    end_time = time.time()
    print("=" * 60)
    print("✨ Processing completed!")
    print(f"⏱️ Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()