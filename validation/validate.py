import difflib


def compare_files(output_path, output_test_path):
    try:
        with open(output_path, 'r') as f_out, open(output_test_path,
                                                   'r') as f_test:
            out_lines = f_out.readlines()
            test_lines = f_test.readlines()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    differences_found = False
    n = len(test_lines)

    for i in range(n):
        out_line = out_lines[i].strip() if i < len(out_lines) else ""
        test_line = test_lines[i].strip()

        if out_line != test_line:
            differences_found = True

            out_tokens = out_line.split()
            test_tokens = test_line.split()

            # Use SequenceMatcher to find differences
            matcher = difflib.SequenceMatcher(None, out_tokens, test_tokens)

            diff_count = 0
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag != 'equal':
                    diff_count += max(i2 - i1, j2 - j1)

            print(f"❌ Line {i+1} differs:")
            print(f"    - Token differences: {diff_count}")
            print("-" * 20)

    if not differences_found:
        print(
            "✅ The first n lines of output.txt match output_test.txt exactly.")


# Example usage
compare_files('../data/output_20b.txt', '../data/output_test.txt')
