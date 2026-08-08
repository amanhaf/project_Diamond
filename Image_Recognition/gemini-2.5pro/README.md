# Image Analysis with Gemini 2.5 Pro

This script uses Google's Gemini 2.5 Pro Vision Language Model to analyze images from the `samples/` directory and generate craft-focused keywords.

## Setup

1. **Activate the virtual environment**:
   ```bash
   source vlm_env/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure your .env file contains your Gemini API key**:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

## Usage

Run the script to analyze all images in the `samples/` directory:

```bash
python analyze_images.py
```

The script will:
- Load your Gemini API key from the .env file
- Process each image in the samples directory
- Generate craft-focused keywords for each image
- Display the results in the terminal

## Output Format

For each image, you'll see:
- The filename being analyzed
- The generated keywords as a comma-separated list


