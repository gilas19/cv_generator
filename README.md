# CV Generator

A Python tool to generate a customized CV (resume) for any job posting. It fetches a job description from a URL, structures it, customizes your CV to match the job, and exports it as a LaTeX and PDF file.

## Features
- Fetches and structures job descriptions from URLs
- Customizes your CV to match job requirements using OpenAI
- Converts the CV to LaTeX using a template
- Compiles the LaTeX to PDF
- Supports both API and manual (clipboard) modes

## Requirements
- Python 3.9+
- [OpenAI Python SDK](https://pypi.org/project/openai/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- LaTeX distribution with `pdflatex` (e.g., MacTeX on macOS)

## Setup
1. Clone this repository or download the files.
2. Install Python dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Install a LaTeX distribution (e.g., MacTeX for macOS, TeX Live for Linux/Windows).
4. Set your OpenAI API key in a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage
Run the script with a job description URL:
```sh
python cv_generator.py <job_url>
```

### Options
- `--output`, `-o`: Output filename (default: CustomizedCV)
- `--api-key`: OpenAI API key (or set in `.env`)
- `--model`: OpenAI model to use (default: gpt-4o-mini)
- `--external-mode`: Use manual clipboard mode (default: True)
- `--save`: Save intermediate files (default: False)
- `--log`: Enable logging output

## File Structure
- `cv_generator.py`: Main script
- `assets/`: Contains templates and CV files
    - `complete_cv.md`: Your full CV in markdown
    - `job_description_template.md`: Template for structuring job descriptions
    - `template.tex`: LaTeX template for the CV
    - `CustomizedCV.tex`/`.pdf`: Output files

## Notes
- The script can run in "external mode" (manual clipboard) if you do not want to use the OpenAI API directly.
- Make sure `pdflatex` is installed and available in your PATH.

## License
MIT License
