

import json

def get_mcq_counts_from_json(file_path):
    """
    Reads a JSON file, calculates the total number of MCQs,
    and the count of MCQs within each section (key).

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        tuple: A tuple containing:
               - int: The total number of MCQs across all sections.
               - dict: A dictionary where keys are section titles and values are
                       the number of MCQs in that section.
               Returns (0, {}) if the file is not found or JSON is invalid.
    """
    total_mcqs = 0
    mcqs_per_section = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for section_title, questions in data.items():
            if isinstance(questions, list):
                count = len(questions)
                mcqs_per_section[section_title] = count
                total_mcqs += count
            # else:
                # Optionally, you could add a warning here if a section value is not a list.
                # print(f"Warning: Section '{section_title}' in JSON is not a list. Skipping.")

    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return 0, {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{file_path}'")
        return 0, {}
    except Exception as e:
        print(f"An unexpected error occurred while processing '{file_path}': {e}")
        return 0, {}

    return total_mcqs, mcqs_per_section

# Fix applied here: using a raw string for the file path
file_to_process = r"C:\Users\abdul\Desktop\csspreparation\prep-pathways-platform\public\urdu-mcqs.json"
# Or, using forward slashes:
# file_to_process = "C:/Users/abdul/Desktop/csspreparation/prep-pathways-platform/public/urdu-mcqs.json"

total_count, section_counts = get_mcq_counts_from_json(file_to_process)

# You can then print the results:
print(f"Total MCQs: {total_count}")
print("MCQs per section:")
for section, count in section_counts.items():
    print(f"  {section}: {count}")