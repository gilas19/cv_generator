"""
CV Generator Script

This script takes a job description URL and generates a customized CV:
1. Fetches job description from URL
2. Converts it to structured format using job description template
3. Generates customized CV based on job requirements
4. Converts to LaTeX using template
5. Exports to PDF

Usage:
    python cv_generator.py <job_url>
"""

import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict
import logging


logger = logging.getLogger("cv_generator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()])


def clipboard(prompt: str) -> str:
    p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
    if p.stdin is not None:
        # Write bytes, not str
        p.stdin.write(bytes(prompt, "utf-8"))
        p.stdin.close()
        p.wait()
    else:
        logger.error("pbcopy stdin is None")
        raise RuntimeError("pbcopy stdin is None")

    input("Press Enter after copying your model output to clipboard... (External mode enabled)")

    p = subprocess.Popen(["pbpaste"], stdout=subprocess.PIPE)
    p.wait()
    if p.stdout is not None:
        data = p.stdout.read()
        # If already str, return as is; if bytes, decode
        if isinstance(data, bytes):
            return data.decode("utf-8")
        return data
    else:
        logger.error("pbpaste stdout is None")
        raise RuntimeError("pbpaste stdout is None")


class CVGenerator:
    def __init__(self, model: str, external_mode: bool, openai_api_key: str = "") -> None:
        """Initialize the CV generator with Responses API key."""
        self.external_mode = external_mode
        if not external_mode:
            from openai import OpenAI

            self.api_key = self._load_key(openai_api_key)
            self.client = OpenAI(api_key=self.api_key)
            self.model = model
        self.workspace_path = Path(__file__).parent / "assets"
        logger.info("CVGenerator initialized. Workspace: %s", self.workspace_path)

    def _load_key(self, openai_api_key: str) -> str:
        """Initialize the CV generator with Responses API key."""
        if not openai_api_key:
            from dotenv import load_dotenv

            load_dotenv()
            openai_api_key = os.getenv("OPENAI_API_KEY") or ""
            if not openai_api_key:
                logger.error("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass it as parameter.")
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass it as parameter.")
        return openai_api_key

    def fetch_and_structure_job(self, url: str) -> str:
        """
        Use OpenAI GPT-4o mini Search Preview to fetch and structure the job description from the URL in one step.
        The job_description_template is included directly in the prompt.
        """
        job_description_template_path = self.workspace_path / "job_description_template.md"
        job_description_template = job_description_template_path.read_text()
        prompt = f"""
You are an expert at analyzing job descriptions. Given the following job posting URL, use your browsing and search capabilities to extract the job description and convert it into the structured format provided below.

Job Posting URL:
{url}

Template Format:
{job_description_template}

Instructions:
- Extract the job title, company name, and location accurately.
- Identify the key responsibilities and requirements.
- Structure the information clearly according to the template.
- If certain information is not available, leave the placeholder or put "Not specified".
- Return only the filled template in markdown format.
"""
        logger.info("Fetching and structuring job description from URL: %s", url)
        if self.external_mode:
            logger.info("External mode enabled: using clipboard for prompt/response.")
            return clipboard(prompt)
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini-search-preview",
                messages=[
                    {"role": "system", "content": "You are an expert at parsing and structuring job descriptions using web search."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1200,
            )
            logger.info("Job description structured successfully.")
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Failed to fetch and structure job description: %s", str(e))
            raise Exception(f"Failed to fetch and structure job description: {str(e)}")

    def generate_customized_cv(self, structured_job_desc: str) -> str:
        """Generate customized CV based on job description and complete CV."""
        complete_cv_path = self.workspace_path / "complete_cv.md"
        complete_cv_content = complete_cv_path.read_text()

        prompt = f"""
You are an expert CV writer. Please customize the following CV to better match the job requirements.

Job Description:
{structured_job_desc}

Current CV:
{complete_cv_content}

Please customize the CV by:
1. Keeping all factual information accurate - DO NOT make up ANY information.
2. Adjusting the profile to highlight relevant skills for this specific role, up to 2 sentences
3. Choose level of education to include based on the job requirements, write GPA only if it is asked for
4. Reordering or emphasizing technical skills that match the job requirements, 5 categories is enough
5. Choose 2 relevant projects that align with the job
6. Choose 2 relevant experiences that align with the job, describe them in up to 2 sentences each
7. Maintaining the same markdown format and structure

Return the customized CV in the exact same markdown format as the original.
"""

        logger.info("Generating customized CV based on structured job description.")
        if self.external_mode:
            logger.info("External mode enabled: using clipboard for prompt/response.")
            return clipboard(prompt)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert CV writer who customizes resumes while maintaining factual accuracy."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2500,
                temperature=0.3,
            )
            logger.info("Customized CV generated successfully.")
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Failed to generate customized CV: %s", str(e))
            raise Exception(f"Failed to generate customized CV: {str(e)}")

    def convert_to_latex(self, customized_cv: str) -> str:
        """Convert customized CV markdown to LaTeX using template."""
        template_path = self.workspace_path / "template.tex"
        template_content = template_path.read_text()

        prompt = f"""
You are an expert at converting CV content from markdown to LaTeX. Please convert the following CV to LaTeX using the provided template.

CV Content:
{customized_cv}

LaTeX Template:
{template_content}

Please:
1. Replace all placeholder content in the template with the actual CV information
2. Maintain the template's formatting and structure
3. Convert markdown formatting to appropriate LaTeX commands
4. Ensure proper LaTeX syntax and escaping of special characters
5. Map the CV sections to the template sections appropriately
6. Handle any special characters or symbols properly for LaTeX

Return only the complete LaTeX document ready for compilation.
"""

        logger.info("Converting customized CV to LaTeX.")
        if self.external_mode:
            logger.info("External mode enabled: using clipboard for prompt/response.")
            return clipboard(prompt)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at converting documents to LaTeX format."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=3000,
                temperature=0.3,
            )
            logger.info("CV converted to LaTeX successfully.")
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Failed to convert CV to LaTeX: %s", str(e))
            raise Exception(f"Failed to convert CV to LaTeX: {str(e)}")

    def compile_latex_to_pdf(self, latex_content: str, output_name: str = "CustomizedCV") -> str:
        logger.info("Compiling LaTeX to PDF: %s", output_name)
        try:
            # Create temporary directory for LaTeX compilation
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Write LaTeX content to file
                tex_file = temp_path / f"{output_name}.tex"
                tex_file.write_text(latex_content, encoding="utf-8")

                # Compile LaTeX to PDF (run twice for proper cross-references)
                for _ in range(2):
                    result = subprocess.run(
                        ["pdflatex", "-interaction=nonstopmode", tex_file.name], cwd=temp_dir, capture_output=True, text=True
                    )
                    if result.returncode != 0:
                        logger.warning("LaTeX compilation warning/error output:\n%s\n%s", result.stdout, result.stderr)

                # Check if PDF was generated
                pdf_file = temp_path / f"{output_name}.pdf"
                if not pdf_file.exists():
                    logger.error("PDF compilation failed - no PDF file generated")
                    raise Exception("PDF compilation failed - no PDF file generated")

                # Copy PDF to workspace
                output_pdf = self.workspace_path / f"{output_name}.pdf"
                output_pdf.write_bytes(pdf_file.read_bytes())

                logger.info("PDF successfully compiled: %s", output_pdf)
                return str(output_pdf)
        except subprocess.CalledProcessError as e:
            logger.error("LaTeX compilation failed: %s", e)
            raise Exception(f"LaTeX compilation failed: {e}")
        except Exception as e:
            logger.error("Failed to compile PDF: %s", str(e))
            raise Exception(f"Failed to compile PDF: {str(e)}")

    def generate_cv_from_url(self, job_url: str, output_name: str = "CustomizedCV", save: bool = False) -> Dict[str, str]:
        logger.info("Starting CV generation from job URL: %s", job_url)
        structured_job = self.fetch_and_structure_job(job_url)
        if save:
            # Save structured job description
            job_output_path = self.workspace_path / f"structured_job_{output_name}.md"
            job_output_path.write_text(structured_job)
            logger.info("Structured job description saved to: %s", job_output_path)

        logger.info("Generating customized CV...")
        customized_cv = self.generate_customized_cv(structured_job)
        if save:
            # Save customized CV markdown
            cv_md_path = self.workspace_path / f"customized_cv_{output_name}.md"
            cv_md_path.write_text(customized_cv)
            logger.info("Customized CV (markdown) saved to: %s", cv_md_path)

        logger.info("Converting to LaTeX...")
        latex_content = self.convert_to_latex(customized_cv)
        # Remove first and last lines from LaTeX content
        if latex_content.startswith("```") and latex_content.endswith("```"):
            latex_content = "\n".join(latex_content.splitlines()[1:-1])
        # Save LaTeX file
        latex_path = self.workspace_path / f"{output_name}.tex"
        latex_path.write_text(latex_content)
        logger.info("LaTeX file saved to: %s", latex_path)

        logger.info("Compiling PDF...")
        pdf_path = self.compile_latex_to_pdf(latex_content, output_name)
        logger.info("PDF generated successfully: %s", pdf_path)

        return {
            "latex_file": str(latex_path),
            "pdf_file": pdf_path,
        }


def main():
    parser = argparse.ArgumentParser(description="Generate customized CV from job description URL")
    parser.add_argument("job_url", required=True, help="URL of the job description")
    parser.add_argument("--output", "-o", default="CustomizedCV", help="Output filename (without extension)")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY environment variable)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--external-mode", default=True, help="Use external mode for manual prompt/response (default: True)")
    parser.add_argument("--save", action="store_true", default=False, help="Save intermediate files (default: False)")
    parser.add_argument("--log", action="store_true", help="Enable logging output")

    args = parser.parse_args()

    if args.log:
        logging.getLogger("cv_generator").disabled = False
        logging.disable(logging.NOTSET)

    try:
        result = subprocess.run(["which", "pdflatex"], capture_output=True)
        if result.returncode != 0:
            logger.error("pdflatex not found. Please install LaTeX distribution (e.g., MacTeX on macOS)")
            sys.exit(1)
        logger.info("pdflatex found at: %s", result.stdout.decode().strip())

        generator = CVGenerator(args.model, args.external_mode, args.api_key)
        results = generator.generate_cv_from_url(args.job_url, args.output, args.save)

        logger.info("CV Generation Complete! Generated files:")
        for desc, path in results.items():
            logger.info("  - %s: %s", desc, path)
    except Exception as e:
        logger.error("Error: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
