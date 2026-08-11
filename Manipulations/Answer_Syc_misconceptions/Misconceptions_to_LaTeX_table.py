import json

INPUT_FILE = "Misconceptions.jsonl"
OUTPUT_FILE = "Misconceptions_LaTeX_table.tex"

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
        open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    outfile.write("\\begin{tabular}{c|p{0.5\\textwidth}|p{0.3\\textwidth}}\n")
    outfile.write("\\textbf{Line} & \\textbf{description} & \\textbf{false_statement} \\\\\n")
    outfile.write("\\hline\n")

    for line_num, line in enumerate(infile, 1):
        data = json.loads(line)

        description = data["description"]
        false_statement = data["false_statement"]

        # Escape LaTeX special characters
        def escape_latex(text):
            replacements = {
                "\\": r"\textbackslash{}",
                "&": r"\&",
                "%": r"\%",
                "$": r"\$",
                "#": r"\#",
                "_": r"\_",
                "{": r"\{",
                "}": r"\}",
                "~": r"\textasciitilde{}",
                "^": r"\textasciicircum{}",
            }
            for char, replacement in replacements.items():
                text = text.replace(char, replacement)
            return text

        description = escape_latex(description)
        false_statement = escape_latex(false_statement)

        outfile.write(f"{line_num} & {description} & {false_statement} \\\\\n")

    outfile.write("\\end{tabular}\n")