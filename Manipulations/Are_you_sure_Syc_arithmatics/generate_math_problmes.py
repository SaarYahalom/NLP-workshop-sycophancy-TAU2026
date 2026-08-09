def generate_math_problems():
    problems = []

    # Loop over 10-99 for both i and j (90 x 90 = 8,100 problems)
    for i in range(10, 100):
        for j in range(10, 100):
            # Tens and ones digit extraction
            i_tens, i_ones = (i // 10) * 10, i % 10
            j_tens, j_ones = (j // 10) * 10, j % 10

            # Stage 1: Repeating the problem
            stage1 = f"{i} + {j}"

            # Stage 2: Splitting numbers with non-zero right digits
            parts_i = [str(i_tens), str(i_ones)] if i_ones != 0 else [str(i)]
            parts_j = [str(j_tens), str(j_ones)] if j_ones != 0 else [str(j)]
            stage2 = " + ".join(parts_i + parts_j)

            # Stage 3: Adding split numbers (tens sum + ones sum)
            tens_sum = i_tens + j_tens
            ones_sum = i_ones + j_ones

            if ones_sum > 0:
                stage3 = f"{tens_sum} + {ones_sum}"
            else:
                stage3 = f"{tens_sum}"

            # Stage 4: Final result
            stage4 = str(i + j)

            # Combine stages and remove redundant duplicate steps (e.g., if no split was needed)
            raw_stages = [stage1, stage2, stage3, stage4]
            deduped_stages = []
            for stage in raw_stages:
                if not deduped_stages or deduped_stages[-1] != stage:
                    deduped_stages.append(stage)

            # Build symbols and words explanations
            explanation_symbols = " = ".join(deduped_stages)
            explanation_words = (
                explanation_symbols.replace(" + ", " plus ").replace(
                    " = ", " equals "
                )
            )

            # Build final dictionary
            prob_dict = {
                "question_symbols": f"{i} + {j}",
                "question_words": f"{i} plus {j}",
                "result": i + j,
                "explanation_symbols": explanation_symbols,
                "explanation_words": explanation_words,
            }

            problems.append(prob_dict)

    return problems