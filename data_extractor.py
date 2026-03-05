import pickle
import os
import sys


def load_pickle_file(file_path):
    """
    Safely loads a pickle file from the given path.
    WARNING: Only load pickle files from trusted sources.
    """
    # Validate file existence
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Validate file extension
    if not file_path.lower().endswith(('.pkl', '.pickle')):
        raise ValueError("Invalid file type. Expected a .pkl or .pickle file.")

    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)  # Load the pickled object
        return data
    except pickle.UnpicklingError:
        raise ValueError("Error: The file is not a valid pickle file.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the pickle file: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_pickle.py <path_to_pickle_file>")
        sys.exit(1)

    pickle_path = sys.argv[1]

    try:
        extracted_data = load_pickle_file(pickle_path)
        print("✅ Pickle file loaded successfully.")
        print("Extracted Data:")
        print(extracted_data)
    except Exception as err:
        print(f"❌ {err}")