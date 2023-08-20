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
file_path = f'{folder}/input_{sufix}_2sE.txt'
detected_encoding = detect_encoding(file_path)
print("Detected encoding:", detected_encoding)

import codecs

def convert_to_utf8(input_file, output_file,source_encoding):
    # Specify the source character encoding (charset) of the input file
    # For example, 'iso-8859-1', 'utf-16', 'windows-1252', etc.
    #source_encoding = 'YOUR_SOURCE_ENCODING_HERE'
    
    # Read the file with the specified encoding
    with codecs.open(input_file, 'r', encoding=source_encoding) as file:
        content = file.read()

    # Write the content back to a new file with UTF-8 encoding
    with open(output_file, 'w', encoding='utf-8') as utf8_file:
        utf8_file.write(content)

# Example usage:
input_file_path = file_path
output_file_path = f'{folder}/output_{sufix}_2sE.txt'
convert_to_utf8(input_file_path, output_file_path,detected_encoding)





