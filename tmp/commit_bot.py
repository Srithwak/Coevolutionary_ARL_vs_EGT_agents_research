#!/usr/bin/env python3
import os
import subprocess
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY not set")
    exit(1)

genai.configure(api_key=API_KEY)

def get_git_diff():
    """Get the staged diff, or working tree diff if none staged."""
    try:
        diff = subprocess.check_output(["git", "diff", "--cached"], text=True)
        if not diff.strip():
            diff = subprocess.check_output(["git", "diff"], text=True)
        return diff
    except subprocess.CalledProcessError as e:
        print("Error getting git diff:", e)
        return ""

def generate_commit_message(diff, style="pick the best", model_name="gemini-2.0-flash"):
    """Generate a commit message with given style using Gemini."""
    if not diff.strip():
        return "No changes to commit."

    # Reset memory: each call is a fresh model instance
    model = genai.GenerativeModel(model_name)

    style_instructions = {
        "very short": "Write a minimal one-line commit message.",
        "detailed": "Write a detailed commit message with context.",
        "conventional": "Follow Conventional Commit style (feat:, fix:, chore:, etc).",
        "pick the best": "Choose the most appropriate style for this change."
    }

    prompt = f"""
You are an assistant that writes Git commit messages.
The user wants the style: {style}.
Instruction: {style_instructions.get(style, style_instructions['pick the best'])}

Analyze the following git diff and generate a commit message:

{diff}
"""
    response = model.generate_content(prompt)
    return response.text.strip()

def main():
    diff = get_git_diff()
    if not diff.strip():
        print("No changes found.")
        return

    print("Select commit message style:")
    print("1) Very short")
    print("2) Detailed")
    print("3) Conventional commit")
    print("4) Pick the best one for me")
    choice = input("Enter choice (1-4): ").strip()

    styles = {
        "1": "very short",
        "2": "detailed",
        "3": "conventional",
        "4": "pick the best"
    }
    style = styles.get(choice, "pick the best")

    message = generate_commit_message(diff, style=style)
    print("\nGenerated Commit Message:\n")
    print(message)

    choice = input("\nUse this commit message? (y/n): ").lower()
    if choice == "y":
        subprocess.run(["git", "commit", "-m", message])

if __name__ == "__main__":
    main()
