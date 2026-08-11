import re
from pathlib import Path

def remove_wikipedia_citations(input_path):
    """
    The wikipedia files have [7] [123] etc inline.
    This removes them
    """
    input_path = Path(input_path)
    output_path = input_path.with_stem(input_path.stem + "_clean")

    text = input_path.read_text(encoding="utf-8")

    # Remove citation markers like [7], [12], [123]
    text = re.sub(r"\[\d+\]", "", text)

    output_path.write_text(text, encoding="utf-8")
    print(f"Saved cleaned file to: {output_path}")


# Example
remove_wikipedia_citations("Wikipedia_Misconceptions.txt")