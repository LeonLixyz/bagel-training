import os
import json
from pathlib import Path

def check_accuracy(root_dir):
    """
    Check accuracy by comparing 'answer' and 'final_answer' fields in JSON files.
    
    Args:
        root_dir (str): Root directory containing folders with JSON files
    
    Returns:
        tuple: (accuracy_percentage, list_of_mismatched_folders)
    """
    total_files = 0
    matches = 0
    mismatched_folders = []
    
    # Walk through all subdirectories
    for folder_path in Path(root_dir).iterdir():
        if folder_path.is_dir():
            folder_name = folder_path.name
            
            # Look for JSON files in this folder
            json_files = list(folder_path.glob("*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Check if both fields exist
                    if 'answer' in data and 'final_answer' in data:
                        total_files += 1
                        answer = str(data['answer']).strip()
                        final_answer = str(data['final_answer']).strip()
                        
                        if answer == final_answer:
                            matches += 1
                        else:
                            mismatched_folders.append(folder_name)
                            print(f"Mismatch in {folder_name}: answer='{answer}' vs final_answer='{final_answer}'")
                
                except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                    print(f"Error processing {json_file}: {e}")
                    continue
    
    # Calculate accuracy
    accuracy = (matches / total_files * 100) if total_files > 0 else 0
    
    return accuracy, mismatched_folders, total_files, matches

def main():
    # Change this path to your folder containing the subfolders
    root_directory = "/home/jovyan/workspace/bagel/MathVista-1000/testmini"
    
    print(f"Checking accuracy in: {root_directory}")
    print("-" * 50)
    
    accuracy, mismatched_folders, total, correct = check_accuracy(root_directory)
    
    print(f"\nResults:")
    print(f"Total files processed: {total}")
    print(f"Correct matches: {correct}")
    print(f"Mismatches: {len(mismatched_folders)}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    if mismatched_folders:
        print(f"\nFolders with mismatches:")
        for folder in sorted(mismatched_folders):
            print(f"  - {folder}")
    else:
        print("\nNo mismatches found!")

if __name__ == "__main__":
    main() 