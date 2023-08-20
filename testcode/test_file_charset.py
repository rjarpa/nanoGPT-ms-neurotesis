import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']


# Read lines from the original file
sufix='99maps'
folder=f'data/microstate_{sufix}_2sE_dataset_pretokenize'

# Example usage:
file_path = f'{folder}/output_{sufix}_2sE.txt'
detected_encoding = detect_encoding(file_path)
print("Detected encoding:", detected_encoding)





