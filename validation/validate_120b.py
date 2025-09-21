def compare_files(output_path, output_test_path):
    with open(output_test_path, 'r') as f_test:
        test_lines = f_test.readlines()

    n = len(test_lines)

    try:
        with open(output_path, 'r') as f_out:
            out_lines = f_out.readlines()
    except FileNotFoundError:
        print(f"❌ Error: The file '{output_path}' was not found.")
        return

    differences_found = False

    for i in range(n):
        if i >= len(out_lines):
            # If output_path is shorter than output_test_path,
            # we consider the remaining lines as different.
            out_line = ""
            test_line = test_lines[i].strip()
            differences_found = True

            test_tokens = test_line.split()
            diff_count = len(test_tokens)

            print(f"❌ Line {i+1} differs (end of output.txt reached):")
            print(f"    - Token differences: {diff_count}")
            print(f"    - output.txt:    (empty line)")
            print(f"    - output_test.txt: {test_line}")
            print("-" * 20)
            continue

        out_line = out_lines[i].strip()
        test_line = test_lines[i].strip()

        if out_line != test_line:
            differences_found = True

            # Split lines into tokens (numbers separated by spaces)
            out_tokens = out_line.split()
            test_tokens = test_line.split()

            # Count token differences
            diff_count = 0
            max_len = max(len(out_tokens), len(test_tokens))

            for j in range(max_len):
                # Use a try-except block to handle lines of different lengths gracefully.
                try:
                    out_token = out_tokens[j]
                except IndexError:
                    out_token = None

                try:
                    test_token = test_tokens[j]
                except IndexError:
                    test_token = None

                if out_token != test_token:
                    diff_count += 1

            print(f"❌ Line {i+1} differs:")
            print(f"    - Token differences: {diff_count}")
            print("-" * 20)

    if not differences_found:
        print(
            "✅ The first n lines of output.txt match output_test.txt exactly.")


# Example usage
compare_files('../data/output_120b.txt', '../data/output_test.txt')
