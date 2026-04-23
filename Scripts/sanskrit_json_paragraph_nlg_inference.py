#!/usr/bin/env python3
"""
Sanskrit-JSON-Paragraph-NLG (Sanskrit JSON Graph Paragraph Natural Language Generation)
Inference script for generating complete paragraphs from Sanskrit JSON graph data.
25-Feb-2026: Prompts edited by Pratibha Rani to make them source language agnostic by removing
word "Sanskrit" from the English and Hindi prompts by replacing with words "input" and "language".
"""

import os
import sys
import re
import json
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
API_KEY = "API_KEY"
MODEL_NAME = "gemini-2.5-flash"
BATCH_CHAR_LIMIT = 30000
NUM_FEW_SHOT_EXAMPLES = 2

# Language-specific prompts for paragraph generation from Sanskrit JSON
LANGUAGE_PROMPTS = {
    "english": (
        "Generate a complete, coherent English paragraph from input language JSON graph data. "
        "The input JSON contains semantic information about multiple sentences that form a paragraph. "
        "CRITICAL RULES - WORK FROM INPUT JSON SEMANTIC STRUCTURE: "
        "- Generate a single, coherent paragraph (not separate sentences) "
        "- The input JSON contains semantic tokens with input language roots, suffixes and their relations (k1, k7, r6, etc.)\n"
        "- Extract meaning from these input language semantic tokens and their relations\n"
        "- Understand input language grammatical concepts like vibhakti, dhatu, pratyaya "
        "- Maintain proper discourse flow and pronoun resolution "
        "- Use natural English expressions and transitions "
        "- STRICTLY preserve all semantic information from the input JSON"
        "- Ensure CONTINUITY: Each sentence should flow naturally into the next "
        "- Ensure FLUIDITY: Use appropriate connectors, transitions, and cohesive devices "
        "- Maintain logical progression and avoid abrupt jumps between ideas "
        "- No JSON notation or technical terms "
        "- Create a paragraph that reads naturally and smoothly "
        "- DO NOT add any information not present in the input JSON data "
        "- DO NOT remove any information present in the input JSON data "
        "- DO NOT create new concepts, objects, or ideas not in the input JSON "
        "- DO NOT add examples, analogies, or explanations not in the input JSON "
        "- Generate ONLY what the input JSON data explicitly contains "
        "- If input JSON data is unclear, stick to what is clearly present "
        "- CRITICAL: ONLY use semantic tokens that are EXPLICITLY present in the input JSON\n"
        "- DO NOT add ANY words, concepts, or ideas that are not in the input JSON semantic tokens\n"
        "- Understand input language semantic roles and translate them appropriately to English "
        "Example: Combine multiple input JSON graph structures into one flowing paragraph in the exact order provided with seamless transitions."
    ),
    "hindi": (
        "आप एक इनपुट JSON से हिंदी वाक्य जनरेटर हैं। आपको इनपुट JSON सेमेंटिक डेटा से हिंदी वाक्य बनाने हैं। "
        "इनपुट JSON डेटा में वाक्यों की सेमेंटिक जानकारी है जैसे शब्द, क्रिया, विशेषण, संबंध आदि। "
        "आपको इस सेमेंटिक जानकारी से हिंदी वाक्य बनाने हैं। "
        "महत्वपूर्ण नियम - केवल इनपुट JSON सामग्री से चिपके रहें: "
        "- इनपुट JSON सेमेंटिक डेटा को पढ़कर हिंदी वाक्य बनाएं "
        "- इनपुट JSON में भाषा धातु, प्रत्यय, विभक्ति के साथ सेमेंटिक टोकन और उनके संबंध (k1, k7, r6, आदि) हैं\n"
        "- इन भाषा सेमेंटिक टोकन और उनके संबंधों से अर्थ निकालें\n"
        "- भाषा व्याकरण की अवधारणाओं को समझें जैसे विभक्ति, धातु, प्रत्यय "
        "- शब्दों, क्रियाओं, विशेषणों को सही तरीके से जोड़ें "
        "- व्याकरण के नियमों का पालन करें "
        "- सभी वाक्यों को क्रमानुसार जोड़कर पैराग्राफ बनाएं "
        "- उचित संयोजक और प्रवाह बनाए रखें "
        "- JSON नोटेशन न दें, सिर्फ हिंदी वाक्य दें "
        "- कड़ाई से इनपुट JSON डेटा से सटीक अर्थ बनाए रखें - कोई जोड़ या घटाव नहीं "
        "- इनपुट JSON डेटा में नहीं है वैसी कोई जानकारी न जोड़ें "
        "- इनपुट JSON डेटा में है वैसी कोई जानकारी न हटाएं "
        "- नए विचार, वस्तु या अवधारणाएं न बनाएं जो इनपुट JSON में नहीं हैं "
        "- उदाहरण, सादृश्य या स्पष्टीकरण न जोड़ें जो इनपुट JSON में नहीं हैं "
        "- केवल वही उत्पन्न करें जो इनपुट JSON डेटा में स्पष्ट रूप से मौजूद है "
        "- यदि इनपुट JSON डेटा अस्पष्ट है, तो स्पष्ट रूप से मौजूद चीज़ों से चिपके रहें "
        "- केवल इनपुट JSON सामग्री - कोई कल्पना नहीं "
        "- इनपुट भाषा सेमेंटिक भूमिकाओं को समझें और उन्हें हिंदी में उचित रूप से अनुवाद करें "
        "उदाहरण: इनपुट JSON डेटा से 'हमारी बदलती पृथ्वी' जैसा वाक्य बनाएं।"
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

def call_gemini_api_batch(api_input_text, api_key, language="hindi", max_retries=3):
    """Makes a single, batched call to the Gemini API with smart quota handling."""
    if genai is None:
        raise RuntimeError("google.generativeai not available")

    genai.configure(api_key=api_key or API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)

    for attempt in range(max_retries):
        wait_for_rate_limit()
        request_times.append(time.time())

        try:
            response = model.generate_content(api_input_text)
            return response.text.strip()

        except Exception as e:
            error_str = str(e).lower()

            # Handle quota errors with retry delay detection
            if "quota" in error_str or "429" in error_str:
                print(f"  - API quota exceeded (attempt {attempt + 1}/{max_retries})")
                
                # Try to extract retry time (e.g., "Please retry in 49.279318361s")
                match = re.search(r"retry in (\d+(\.\d+)?)s", error_str)
                if match:
                    delay = float(match.group(1)) + 5  # Add safety buffer
                    print(f"    → Waiting for {int(delay)} seconds before retrying...")
                    time.sleep(delay)
                else:
                    print("    → Waiting 60 seconds (default cooldown)...")
                    time.sleep(60)

                continue

            elif "rate limit" in error_str:
                print("  - Rate limit reached, waiting 60 seconds...")
                time.sleep(60)
                continue

            else:
                print(f"  - API error (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(5)
                continue

    print("  - Failed after maximum retries.")
    return ""

def parse_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse Sanskrit JSON file and extract sentence blocks in order."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ensure list
    if isinstance(data, dict):
        data = [data]
    
    items = []
    
    # Handle list of structures
    for i, structure in enumerate(data):
        if isinstance(structure, dict):
            # Extract usr_id and text (for converted Sanskrit USR data)
            sent_id = structure.get('usr_id', structure.get('sent_id', f'structure_{i}'))
            original_sentence = structure.get('text', structure.get('original', ''))
            
            # Create masked version
            masked_structure = structure.copy()
            if "text" in masked_structure:
                masked_structure["text"] = "[masked]"
            if "original" in masked_structure:
                masked_structure["original"] = "[masked]"
            
            if original_sentence:
                items.append({
                    'id': sent_id,
                    'original': original_sentence,
                    'api_content': json.dumps(masked_structure, ensure_ascii=False, indent=2)
                })
    
    # Sort by ID to maintain order
    items.sort(key=lambda x: x['id'])
    return items

def create_paragraph_prompt(language: str, items: List[Dict[str, Any]], few_shot_examples: List[Dict[str, Any]] = None) -> str:
    """Create a prompt for paragraph generation from Sanskrit JSON."""
    prompt = LANGUAGE_PROMPTS[language] + "\n\n"
    
    # Add few-shot examples if available
    if few_shot_examples:
        prompt += "Examples:\n"
        for i, example in enumerate(few_shot_examples, 1):
            prompt += f"Example {i}:\n"
            for item in example['items']:
                prompt += f"{item['api_content']}\n"
            prompt += f"Generated Paragraph: [masked]\n\n"
        prompt += "--- GENERATE ---\n\n"
    
    # Add target items with more specific instructions
    prompt += f"CRITICAL: Generate a complete paragraph from the following {len(items)} input JSON structures. "
    prompt += "You MUST translate the EXACT content from each input JSON structure in the given order. "
    prompt += "Do NOT add any information that is not present in the input JSON structures.\n\n"
    
    for i, item in enumerate(items, 1):
        prompt += f"Structure {i} (ID: {item['id']}):\n{item['api_content']}\n\n"
    
    prompt += f"CRITICAL INSTRUCTIONS:\n"
    prompt += f"- Study the input JSON semantic data in each structure above\n"
    prompt += f"- Extract the meaning from the input semantic information (words, verbs, adjectives, relations)\n"
    prompt += f"- Generate proper {language} sentences from the input semantic data\n"
    prompt += f"- Follow the EXACT order of structures (1, 2, 3, ...)\n"
    prompt += f"- Combine all {len(items)} structures into one coherent paragraph\n"
    prompt += f"- Do NOT add any information not present in the input JSON structures\n"
    prompt += f"- Do NOT skip any structures\n"
    prompt += f"- Use appropriate connectors to make the paragraph flow naturally\n"
    prompt += f"- IMPORTANT: Replace [masked] with the actual {language} sentence you generate from input JSON data\n"
    prompt += f"- Do NOT output [masked] or any JSON notation in your response\n"
    prompt += f"- Generate ONLY the final {language} paragraph\n"
    prompt += f"- WORK DIRECTLY FROM THE INPUT JSON SEMANTIC STRUCTURE\n"
    prompt += f"- The Input JSON contains language semantic tokens with roots, suffixes, and grammatical relations\n"
    prompt += f"- Extract meaning from these language semantic tokens and their relations (k1, k7, r6, rblsk, etc.)\n"
    prompt += f"- If input JSON has interrogative, generate a question\n"
    prompt += f"- If input JSON has affirmative, generate a statement\n"
    prompt += f"- The semantic relations (k1, k7, r6, etc.) show how language words connect\n"
    prompt += f"- CRITICAL: ONLY use semantic tokens that are EXPLICITLY present in the input JSON\n"
    prompt += f"- DO NOT add ANY words, concepts, or ideas that are not in the input JSON semantic tokens\n"
    prompt += f"- If a semantic token is not in the input JSON, DO NOT mention it in the output\n\n"
    prompt += f"Generate the paragraph:"
    return prompt

def wrap_text(text: str, width: int = 80) -> str:
    """Wrap text to specified width for better readability."""
    import textwrap
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)

def process_file(input_file: str, output_file: str, language: str, mode: str = "zero_shot", train_file: str = None) -> None:
    """Process Sanskrit JSON file and generate paragraph output in safe batches."""
    print(f"Processing {input_file} for paragraph generation...")

    # Parse input file
    items = parse_json_file(input_file)
    if not items:
        print(f"  - No valid items found in {input_file}")
        return

    print(f"  - Found {len(items)} structures to combine into paragraph")

    # Prepare few-shot examples if needed
    few_shot = None
    if mode == "few_shot" and train_file and os.path.exists(train_file):
        train_items = parse_json_file(train_file)
        if len(train_items) >= 3:
            import random
            num_example_items = min(random.randint(3, 5), len(train_items))
            example_items = random.sample(train_items, num_example_items)
            example_items.sort(key=lambda x: x['id'])
            few_shot = [{'items': example_items}]
            print(f"  - Using {len(example_items)} few-shot example items")
        else:
            print("  - Not enough train examples; falling back to zero-shot")

    # Split into batches based on character size
    current_batch = []
    batch_char_count = 0
    batch_id = 1
    paragraphs = []

    for item in items:
        item_str = json.dumps(item, ensure_ascii=False)
        if batch_char_count + len(item_str) > BATCH_CHAR_LIMIT:
            # Process current batch
            print(f"  - Processing batch {batch_id} with {len(current_batch)} structures...")
            prompt = create_paragraph_prompt(language, current_batch, few_shot)
            response_text = call_gemini_api_batch(prompt, None, language)
            paragraphs.append(response_text.strip() if response_text else "[NO RESPONSE]")

            # Sleep to respect quota
            print("  - Waiting 30 seconds before next batch...")
            time.sleep(60)

            # Reset for next batch
            current_batch = []
            batch_char_count = 0
            batch_id += 1

        current_batch.append(item)
        batch_char_count += len(item_str)

    # Process remaining batch
    if current_batch:
        print(f"  - Processing batch {batch_id} with {len(current_batch)} structures...")
        prompt = create_paragraph_prompt(language, current_batch, few_shot)
        response_text = call_gemini_api_batch(prompt, None, language)
        paragraphs.append(response_text.strip() if response_text else "[NO RESPONSE]")

    # Combine all partial paragraphs
    final_paragraph = "\n\n".join(paragraphs)

    # Write to output file
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        wrapped_paragraph = wrap_text(final_paragraph, width=80)
        f.write(wrapped_paragraph + "\n")

    print(f"  - Combined {len(paragraphs)} batches into final paragraph in {output_file}")


# def main():
#     parser = argparse.ArgumentParser(description="Sanskrit-JSON-Paragraph-NLG Inference Script")
#     parser.add_argument("input_file", help="Input Sanskrit JSON file path")
#     parser.add_argument("-o", "--output", help="Output file path")
#     parser.add_argument("-l", "--language", default="english", 
#                        choices=["english", "hindi"],
#                        help="Target language (default: english)")
#     parser.add_argument("-m", "--mode", default="zero_shot",
#                        choices=["zero_shot", "few_shot"],
#                        help="Inference mode (default: zero_shot)")
    
#     args = parser.parse_args()
    
#     if not os.path.exists(args.input_file):
#         print(f"Error: Input file {args.input_file} does not exist")
#         return
    
#     if not args.output:
#         base_name = os.path.splitext(os.path.basename(args.input_file))[0]
#         args.output = f"{base_name}_{args.language}_sanskrit_gemini_json.txt"
    
#     process_file(args.input_file, args.output, args.language, args.mode)

# if __name__ == "__main__":
#     main()
def main():
    import glob
    parser = argparse.ArgumentParser(description="Sanskrit-JSON-Paragraph-NLG Inference Script")
    parser.add_argument("json_folder", help="Folder containing input Sanskrit JSON files (e.g., ./json)")
    parser.add_argument("-o", "--output_folder", default="output",
                       help="Folder where output paragraphs will be saved (default: ./output)")
    parser.add_argument("-l", "--language", default="english",
                       choices=["english", "hindi"],
                       help="Target language (default: english)")
    parser.add_argument("-m", "--mode", default="zero_shot",
                       choices=["zero_shot", "few_shot"],
                       help="Inference mode (default: zero_shot)")

    args = parser.parse_args()
    json_folder = args.json_folder
    output_folder = args.output_folder

    if not os.path.exists(json_folder):
        print(f"❌ Folder not found: {json_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    # 🧩 DEBUG: Show what files exist
    print(f"\n📂 Checking folder: {json_folder}")
    all_files = os.listdir(json_folder)
    print(f"Found {len(all_files)} total files:")
    for f in all_files:
        print("  -", f)

    # Filter for prefix match
    json_files = [
        f for f in all_files
        if f.startswith("test3_json") and f.endswith(".json")
    ]
    print(f"\n🎯 Matched {len(json_files)} files starting with '01_nov_25':")
    for f in json_files:
        print("  ✅", f)

    if not json_files:
        print("⚠️ No matching files found! (check filename prefix or folder path)")
        return

    for filename in sorted(json_files):
        input_path = os.path.join(json_folder, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, f"{base_name}_gemini_english_json.txt")

        print(f"\n🔹 Processing: {filename}")
        print(f"   → Input: {input_path}")
        print(f"   → Output: {output_path}")

        try:
            process_file(input_path, output_path, args.language, args.mode)
            if os.path.exists(output_path):
                print(f"✅ File created: {output_path}")
            else:
                print(f"⚠️ No output generated for {filename}")
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print("\n✅ Done. Check your output folder.")
if __name__ == "__main__":
    main()
