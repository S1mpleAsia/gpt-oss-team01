def compare_files(output_path, output_test_path):
    with open(output_test_path, 'r') as f_test:
        test_lines = f_test.readlines()

    n = len(test_lines)

    with open(output_path, 'r') as f_out:
        out_lines = f_out.readlines()

    # Compare first n lines
    match = out_lines[:n] == test_lines

    if match:
        print(
            "✅ The first n lines of output.txt match output_test.txt exactly.")
    else:
        print("❌ The files differ in the first n lines.")
        for i in range(n):
            if out_lines[i] != test_lines[i]:
                print(f"Line {i+1} differs:")
                print(f"  output.txt     : {out_lines[i].rstrip()}")
                print(f"  output_test.txt: {test_lines[i].rstrip()}")
                break


# Example usage
compare_files('../data/output_120b.txt', '../data/output_test.txt')
