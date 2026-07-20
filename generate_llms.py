import os

def read_docs(directory):
    """
    Reads all documentation files in the given directory.

    Args:
        directory (str): The directory containing the documentation files.

    Returns:
        str: The concatenated content of all documentation files.
    """
    docs_content = ""
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                docs_content += file.read()
    return docs_content

def summarize_project(docs_content):
    """
    Summarizes the project based on the given documentation content.

    Args:
        docs_content (str): The concatenated content of all documentation files.

    Returns:
        str: A summary of the project.
    """
    # This is a simple implementation that extracts the first 1000 characters
    # You can improve this by using a natural language processing library
    return docs_content[:1000]

def write_summary_to_file(summary):
    """
    Writes the summary to a file named llms.txt.

    Args:
        summary (str): The summary of the project.
    """
    with open("llms.txt", 'w') as file:
        file.write(summary)

def main():
    directory = "docs"
    docs_content = read_docs(directory)
    summary = summarize_project(docs_content)
    write_summary_to_file(summary)

if __name__ == "__main__":
    main()
