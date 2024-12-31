import string

class ASREncoderDecoder:
    def __init__(self):
        """
        Initializes the character-to-index and index-to-character mappings.
        """
        self.characters = list(string.ascii_lowercase) + [' ']
        self.char_to_index = {char: idx + 1 for idx, char in enumerate(self.characters)}
        self.char_to_index['<blank>'] = 0
        self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}
    
    def encode(self, text):
        """
        Encodes a given text string into a list of integers based on the character-to-index mapping.

        Parameters:
            text (str): The input text to encode.

        Returns:
            List[int]: A list of integers representing the encoded text.
        """
        text = text.lower()
        encoded = []
        for char in text:
            if char in self.char_to_index:
                encoded.append(self.char_to_index[char])
            elif char == "'":
                # Treat apostrophes as spaces
                encoded.append(self.char_to_index[' '])
        return encoded
    
    def decode(self, indices):
        """
        Decodes a list of integers back into a text string based on the index-to-character mapping.

        Parameters:
            indices (List[int]): The list of integers to decode.

        Returns:
            str: The decoded text string.
        """
        chars = []
        for idx in indices:
            if idx in self.index_to_char:
                char = self.index_to_char[idx]
                if char != '<blank>':  # Dont pay attention to epsilon
                    chars.append(char)
        return ''.join(chars)
    def __len__(self):
        return len(self.char_to_index)
    



if __name__ == "__main__":
    # Example
    encoder_decoder = ASREncoderDecoder()
    print(f'The number of classes will be: {len(encoder_decoder)}')
    sample_text = "Hello World!"
    encoded = encoder_decoder.encode(sample_text)
    print(f"Encoded '{sample_text}': {encoded}")
    encoded_sample = [8, 5, 12, 12, 15, 27, 0, 23, 15, 18, 12, 4]
    decoded = encoder_decoder.decode(encoded_sample)
    print(f"Decoded {encoded_sample}: '{decoded}'")
