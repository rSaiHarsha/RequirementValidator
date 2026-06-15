import re
import os

filepath = 'c:\\Users\\ASUS\\Documents\\project\\RequirementValidator\\Analysis\\quality_analyser.py'

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip_next = False

for i, line in enumerate(lines):
    if line.strip() == 'max_retries = 3':
        continue
    
    if line.strip() == 'for attempt in range(max_retries):':
        continue
    
    # Check if we are inside the try block of correct_single_requirement that was indented
    # We unindent by 4 spaces between line 143 and 392 (which is after for attempt in range...)
    # We can just check the line numbers or start/end strings.
    
    # Actually, a simple state machine:
    pass

# A simpler way is to just read the file, find the block to unindent, and do it.
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace "max_retries = 3\n"
content = re.sub(r'    max_retries = 3\n', '', content)
# Replace "for attempt in range(max_retries):\n"
content = re.sub(r'        for attempt in range\(max_retries\):\n', '', content)

# Now we need to unindent the block that was inside the for loop.
# The block starts right after the for loop and ends right before except Exception as e:
# Let's find the start and end of that block.

start_str = '            system_prompt = """'
end_str = '            return index, r, r.content, full_corrected, full_corrected\n'

start_idx = content.find(start_str)
end_idx = content.find(end_str) + len(end_str)

if start_idx != -1 and end_idx != -1:
    block = content[start_idx:end_idx]
    
    # We also need to remove the unused `was_split` logic.
    was_split_code = """                # Track if it was successfully split so we don't punish it in validation
                was_split = split_required or (isinstance(corrected_requirements, list) and len(corrected_requirements) > 1)"""
    block = block.replace(was_split_code, "")
    
    # Unindent block by 4 spaces
    unindented_block = "\n".join(
        line[4:] if line.startswith("    ") else line 
        for line in block.split("\n")
    )
    
    content = content[:start_idx] + unindented_block + content[end_idx:]

# Add correct_requirements at the end of the file or after analyze_requirements
# We will just append it if not already there
correct_reqs_code = """
def correct_requirements(requirements: List[Requirement], llm=None, progress_callback=None, rag=None, mode="single", selected_collections=None) -> List[Dict[str, Any]]:
    if not llm:
        raise ValueError("LLMManager is required for quality analysis.")

    total = len(requirements)
    if total == 0:
        return []

    rag_contexts = [""] * total
    if rag:
        try:
            full_reqs = [r.content for r in requirements]
            rag_contexts = rag.query_batch(full_reqs, collection_name=selected_collections, top_k=2)
        except Exception:
            pass

    correction_data = [None] * total
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(correct_single_requirement, i, r, llm, rag, rag_contexts[i], selected_collections): i 
            for i, r in enumerate(requirements)
        }
        
        completed_count = 0
        for future in as_completed(futures):
            index, result = future.result()
            correction_data[index] = {
                "ID": result[1].name,
                "Original Requirement": result[1].content,
                "Corrected Requirement": result[4]
            }
            completed_count += 1
            if progress_callback:
                progress_callback(completed_count, total)
                
    return correction_data
"""

if "def correct_requirements" not in content:
    content = content.replace("def generate_markdown_report", correct_reqs_code + "\ndef generate_markdown_report")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated quality_analyser.py successfully.")
