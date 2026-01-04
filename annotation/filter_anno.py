import json
import re

# 1. Config: Input and Output filenames
input_file = '/home/manish/Desktop/projects/video_Generation/vg_finetune_2/annotation/white_board_annotation/test_video_anno.jsonl'
output_file = '/home/manish/Desktop/projects/video_Generation/vg_finetune_2/annotation/white_board_annotation/cleaned_video_anno.jsonl'

# 2. The Correction Dictionary (Regex Patterns -> Correct Math Term)
# These specific patterns were identified from your uploaded file.
corrections = {
    r"(?i)my dreams": "Magnet Brains",
    r"(?i)bottle engineer": "linear",
    r"(?i)turmeric": "linear",
    r"(?i)airline reference": "line reference",
    r"(?i)fennel": "finite",
    r"(?i)dubable bits": "available points",
    r"(?i)wedding": "bending",
    r"(?i)unprofessional": "unbroken",
    r"(?i)school end": "scaled end",
    r"(?i)customer": "cluster",  # or "curve" depending on context
    r"(?i)faluda": "4",
    r"(?i)protein": "14",
    r"(?i)regional forest officer": "original data figures",
    r"(?i)alexis": "X-axis",
    r"(?i)virus": "Y-axis",
    r"(?i)violence": "Y-axis",
    r"(?i)rudra": "Draw a",
    r"(?i)hukumat": "coordinate",
    r"(?i)loot of": "value of",
    r"(?i)sallu mian": "value in",  # "Sallu Mian" likely misheard "value in"
    r"(?i)farindia": "quadrants",
    r"(?i)husband will get me too call": "let us get to the recall",
}

def clean_text(text):
    # Apply all corrections
    for error, fix in corrections.items():
        text = re.sub(error, fix, text)
    
    # General cleanup: Capitalize sentences, fix punctuation spacing
    text = text.replace(" ,", ",").replace(" .", ".")
    text = text.strip()
    return text

# 3. Process the file
print(f"Processing {input_file}...")
with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    for line_num, line in enumerate(infile):
        if not line.strip(): continue
        
        try:
            data = json.loads(line)
            
            # Keep original text for reference if needed, but overwrite 'text' for training
            original_text = data.get('text', '')
            cleaned = clean_text(original_text)
            
            # Only update if changes were made
            if original_text != cleaned:
                data['text'] = cleaned
                # Optional: Add a flag so you know it was auto-cleaned
                data['is_cleaned'] = True
            
            # Write back to JSONL
            outfile.write(json.dumps(data) + '\n')
            
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON on line {line_num + 1}")

print(f"Done! Cleaned data saved to: {output_file}")
print("Example correction:")
print("Before: 'make and bottle engineer graph'")
print("After:  'make and linear graph'")