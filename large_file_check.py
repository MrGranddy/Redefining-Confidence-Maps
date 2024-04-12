import os
import fnmatch

def read_gitignore(gitignore_path):
    """Read and parse .gitignore, returning a list of patterns."""
    patterns = []
    try:
        with open(gitignore_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):
                    patterns.append(line)
    except FileNotFoundError:
        print("No .gitignore file found.")
    return patterns

def is_ignored(path, patterns):
    """Determine if the file should be ignored based on .gitignore patterns."""
    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern):
            return True
    return False

def find_large_files(directory, ignore_patterns, min_size_mb):
    """Recursively find large files not ignored by .gitignore patterns."""
    min_size_bytes = min_size_mb * 1024 * 1024  # Convert MB to bytes
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not is_ignored(file_path, ignore_patterns):
                if os.path.getsize(file_path) > min_size_bytes:
                    print(file_path)

def main():
    # Adjust these parameters as necessary
    current_directory = os.getcwd()
    gitignore_path = os.path.join(current_directory, '.gitignore')
    min_size_mb = 2 # Files larger than 10 MB

    ignore_patterns = read_gitignore(gitignore_path)
    find_large_files(current_directory, ignore_patterns, min_size_mb)

if __name__ == "__main__":
    main()
