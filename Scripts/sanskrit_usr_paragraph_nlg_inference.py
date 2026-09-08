#!/usr/bin/env python3
"""
Sanskrit-USR-Paragraph-NLG (Sanskrit Universal Semantic Representation Paragraph Natural Language Generation)
"""

import os
import sys
import re
import argparse
import time
from collections import deque
from typing import List, Dict, Any

# Try to import google.generativeai
try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("Warning: google.generativeai not available. Install with: pip install google-generativeai")

# Configuration
# Read API key from environment variable
API_KEY = os.environ.get("GEMINI_API_KEY", "")
BATCH_CHAR_LIMIT = int(os.environ.get("BATCH_CHAR_LIMIT", "30000"))

# Language-specific prompts for paragraph generation from Sanskrit USR
LANGUAGE_PROMPTS = {
    "english": (
        "You are an English paragraph generator from USR semantic data input. "
        "The input USR contains semantic information about sentences. "
        "You need to generate natural, flowing English sentences from this input language semantic data. "
        "CRITICAL RULES - WORK FROM INPUT USR SEMANTIC STRUCTURE: "
        "- Read input USR semantic data and generate natural English sentences "
        "- The input USR contains semantic tokens with language roots, suffixes, and grammatical relations "
        "- Extract meaning from these input language semantic tokens and their relations (like k1, k7, r6, etc.) "
        "- Understand input language grammatical concepts like vibhakti, dhatu, pratyaya "
        "- Combine words, verbs, adjectives correctly based on input language USR semantic structure "
        "- Follow English grammar rules in output "
        "- Join all sentences in order to form a coherent paragraph "
        "- Use appropriate connectors and maintain natural flow "
        "- Do not output USR notation, only natural English sentences "
        "- Replace #[masked] with actual English sentences you generate from input USR data "
        "- Make the text sound natural and conversational, not formal or instructional "
        "- CRITICAL: Do NOT translate '$addressee' literally - ignore it or make the sentence natural "
        "- Avoid repetitive addressing like 'Respected addressee' - make it flow naturally "
        "- Generate natural sentences as if explaining to a general audience "
        "- STRICTLY preserve the EXACT meaning from the input USR data - NO ADDITIONS OR REMOVALS "
        "- DO NOT add any information not present in the input USR data "
        "- DO NOT remove any information present in the input USR data "
        "- DO NOT create new concepts, objects, or ideas not in the input USR "
        "- DO NOT add examples, analogies, or explanations not in the input USR "
        "- DO NOT add any new content, facts, or information "
        "- DO NOT create scenarios, situations, or contexts not in the input USR "
        "- DO NOT add descriptive details not present in the input USR "
        "- Only add a few words here and there for fluency, nothing more "
        "- If input USR data is unclear, stick to what is clearly present "
        "- Generate ONLY what the input USR data explicitly contains "
        "- NO HALLUCINATION - ONLY INPUT USR CONTENT "
        "- Understand input language semantic roles and translate them appropriately to English "
        "Example: Generate natural English sentences from input language USR semantic structure."
    )
}

# --- Rate Limiting & API Call ---
request_times = deque()
MAX_REQUESTS_PER_MIN = 15

def wait_for_rate_limit():
    """Pauses execution if the number of recent requests exceeds the limit."""
    while len(request_times) >= MAX_REQUESTS_PER_MIN:
        time_since_oldest = time.time() - request_times[0]
        if time_since_oldest < 60:
            sleep_time = (60 - time_since_oldest) + 2
            print(f"  - Rate limit reached. Pausing for {int(sleep_time)} seconds...")
            time.sleep(sleep_time)
        request_times.popleft()

def call_gemini_api_batch(api_input_text, language="english", model_name=None, max_retries=3):
    """Makes a single, batched call to the Gemini API with retry logic."""
    if genai is None:
        raise RuntimeError("google.generativeai not available")

    effective_model = model_name or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    for attempt in range(max_retries):
        wait_for_rate_limit()

        try:
            # ensure API key present
            if not API_KEY:
                raise RuntimeError("Missing GEMINI_API_KEY environment variable")

            genai.configure(api_key=API_KEY)
            model = genai.GenerativeModel(effective_model)

            request_times.append(time.time())

            response = model.generate_content(api_input_text)
            return response.text.strip()

        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "429" in error_str:
                print(f"  - API quota exceeded (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(4)
                    continue
                return ""
            else:
                print(f"  - API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(4)
                    continue
                return ""

    return ""

def parse_usr_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse Sanskrit USR file and extract sentence blocks in order."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into sentence blocks (handle both sent_id and segment_id)
    sent_blocks = re.split(r'(?=<sent_id=|segment_id=)', content.strip())
    sent_blocks = [block.strip() for block in sent_blocks if block.strip() and not block.startswith('</')]
    
    items = []
    for block in sent_blocks:
        # Extract sent_id or segment_id
        sent_id_match = re.search(r'(sent_id|segment_id)=([^>]+)>', block)
        if not sent_id_match:
            continue
        
        sent_id = sent_id_match.group(2)
        
        # Extract original sentence (line starting with #)
        lines = block.split('\n')
        original_sentence = ""
        masked_block = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                # This is the Sanskrit sentence line - mask it
                original_sentence = line.replace('#', '').strip()
                masked_block.append('#[masked]')
            else:
                masked_block.append(line)
        
        if original_sentence:
            items.append({
                'id': sent_id,
                'original': original_sentence,
                'usr_block': '\n'.join(masked_block)
            })
    
    # Sort by ID to maintain order
    items.sort(key=lambda x: x['id'])
    return items

def create_paragraph_prompt(items: List[Dict[str, Any]]) -> str:
    """Create a prompt for paragraph generation from Sanskrit USR (English only)."""
    prompt = LANGUAGE_PROMPTS["english"] + "\n\n"
    
    # (few-shot removed) — zero-shot English-only prompt
    
    # Add target items with more specific instructions
    prompt += f"CRITICAL: Generate a complete paragraph from the following {len(items)} input USR segments. "
    prompt += "You MUST translate the EXACT content from each input USR segment in the given order. "
    prompt += "Do NOT add any information that is not present in the input USR segments.\n\n"
    
    for i, item in enumerate(items, 1):
        prompt += f"Segment {i} (ID: {item['id']}):\n{item['usr_block']}\n\n"
    
    prompt += f"CRITICAL INSTRUCTIONS:\n"
    prompt += f"- Study the input language USR semantic data in each segment above\n"
    prompt += f"- Extract the meaning from the input language semantic information (words, verbs, adjectives, relations)\n"
    prompt += f"- Generate proper English sentences from the input language semantic data\n"
    prompt += f"- Follow the EXACT order of segments (1, 2, 3, ...)\n"
    prompt += f"- Combine all {len(items)} segments into one coherent paragraph\n"
    prompt += f"- Do NOT add any information not present in the input USR segments\n"
    prompt += f"- Do NOT skip any segments\n"
    prompt += f"- Use appropriate connectors to make the paragraph flow naturally\n"
    prompt += f"- IMPORTANT: Replace #[masked] with the actual English sentence you generate from input USR data\n"
    prompt += f"- Do NOT output #[masked] or any USR notation in your response\n"
    prompt += f"- Generate ONLY the final English paragraph\n"
    prompt += f"- WORK DIRECTLY FROM THE INPUT LANGUAGE USR SEMANTIC STRUCTURE\n"
    prompt += f"- The input USR contains input language semantic tokens with roots, suffixes, and grammatical relations\n"
    prompt += f"- Extract meaning from these input language semantic tokens and their relations (k1, k7, r6, rblsk, etc.)\n"
    prompt += f"- If input USR has %interrogative, generate a question\n"
    prompt += f"- If input USR has %affirmative, generate a statement\n"
    prompt += f"- Pay attention to the main verb (marked with 0:main) and its arguments\n"
    prompt += f"- Follow the semantic relations to understand the sentence structure\n"
    prompt += f"- For questions: $kim means 'what' in input language, look for the object being asked about\n"
    prompt += f"- The semantic relations (k1, k7, r6, etc.) show how input language words connect\n"
    prompt += f"- CRITICAL: ONLY use semantic tokens that are EXPLICITLY present in the input USR\n"
    prompt += f"- DO NOT add ANY words, concepts, or ideas that are not in the input USR semantic tokens\n"
    prompt += f"- If a semantic token is not in the input USR, DO NOT mention it in the output\n"
    prompt += f"- Do not add your own interpretations or explanations\n"
    prompt += f"- Do not add extra words, phrases, or information not in the input USR\n"
    prompt += f"- Do not change questions into statements or vice versa\n"
    prompt += f"- Do not change time references\n"
    prompt += f"- Do not change action descriptions\n"
    prompt += f"- Do not omit any semantic information from the input USR\n"
    prompt += f"- Generate from input USR semantic structure, then make it fluent\n"
    prompt += f"- Preserve ALL semantic information from the input USR\n\n"
    prompt += f"Generate the paragraph:"
    return prompt

def wrap_text(text: str, width: int = 80) -> str:
    """Wrap text to specified width for better readability."""
    import textwrap
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)

def process_file(input_file: str, output_file: str, model_name: str = None) -> None:
    """Process a single Sanskrit USR file and generate an English paragraph output."""
    print(f"Processing {input_file} for paragraph generation...")

    # Parse input file
    items = parse_usr_file(input_file)
    if not items:
        print(f"  - No valid items found in {input_file}")
        return

    print(f"  - Found {len(items)} segments to combine into paragraph")

    # Create prompt (English only)
    prompt = create_paragraph_prompt(items)

    # Check prompt length
    if len(prompt) > BATCH_CHAR_LIMIT:
        print(f"  - Warning: Prompt is {len(prompt)} characters (limit: {BATCH_CHAR_LIMIT})")
        print("  - Consider processing in smaller batches")

    # Call API
    print("  - Generating paragraph in English...")
    response_text = call_gemini_api_batch(prompt, model_name=model_name)

    if not response_text:
        print(f"  - No response received")
        generated_paragraph = "[NO RESPONSE]"
    else:
        generated_paragraph = response_text.strip()

    # Write results with word wrapping
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        wrapped_paragraph = wrap_text(generated_paragraph, width=80)
        f.write(wrapped_paragraph + "\n")

    print(f"  - Generated paragraph in {output_file}")



def main():
    parser = argparse.ArgumentParser(description="Sanskrit-USR-Paragraph-NLG Inference Script (Batch Mode)")
    parser.add_argument("usr_folder", help="Folder containing Sanskrit USR input files (e.g., ./InputDataSanskrit/usr)")
    parser.add_argument("-o", "--output_folder", default="output_usr",
                       help="Folder to save generated paragraph outputs (default: ./output_usr)")
    parser.add_argument("-m", "--model", default="gemini-2.5-flash",
                       choices=["gemini-2.5-pro", "gemini-2.0-flash","gemini-2.5-flash","gemini-1.5-flash", "gemini-1.5-pro"],
                       help="Gemini model to use (default: gemini-2.0-flash)")

    args = parser.parse_args()

    usr_folder = args.usr_folder
    output_folder = args.output_folder

    if not os.path.exists(usr_folder):
        print(f"❌ Input folder '{usr_folder}' not found.")
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)

    # --- Find all Sanskrit USR files (*.usr, *.txt) ---
    usr_files = [
        f for f in os.listdir(usr_folder)
        if (f.endswith(".usr") or f.endswith(".txt"))
    ]

    if not usr_files:
        print(f"⚠️ No matching .usr/.txt files found in '{usr_folder}'.")
        return

    print(f"📂 Found {len(usr_files)} Sanskrit USR files.")
    print(f"💾 Output will be saved in: {output_folder}\n")

    # --- Configure API once ---
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        print("Error: API key not set. Please add it to the script or environment variable.")
        sys.exit(1)

    genai.configure(api_key=API_KEY)

    # --- Process each USR file ---
    for filename in sorted(usr_files):
        input_path = os.path.join(usr_folder, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, f"{base_name}_gemini_english_usr.txt")

        print(f"🔹 Processing: {filename}")
        print(f"   → Input:  {input_path}")
        print(f"   → Output: {output_path}")

        try:
            process_file(input_path, output_path, args.model)
            if os.path.exists(output_path):
                print(f"✅ Successfully created: {output_path}\n")
            else:
                print(f"⚠️ No output generated for {filename}\n")
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}\n")

    print(f"🎉 All matching files processed successfully.\nOutputs saved in '{output_folder}'.")

if __name__ == "__main__":
    main()
