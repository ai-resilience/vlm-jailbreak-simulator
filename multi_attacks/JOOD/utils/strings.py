"""
JOOD
Copyright (c) 2025-present NAVER Corp.
Apache License v2.0
"""
import hashlib

def string_to_hash(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**32)

def interleave_words(word1, word2):
    mixed_word = []
    len1, len2 = len(word1), len(word2)
    total_len = len1 + len2

    index1, index2 = 0, 0
    for i in range(total_len):
        if index1 < len1 and (index2 >= len2 or (index1 / len1) <= (index2 / len2)):
            mixed_word.append(word1[index1])
            index1 += 1
        elif index2 < len2:
            mixed_word.append(word2[index2])
            index2 += 1

    return ''.join(mixed_word)

def concat_words(word1, word2):
    return word1 + word2

def interleave_words_vertically(word1, word2):
    # Determine the length of the longer word
    max_len = max(len(word1), len(word2))
    
    # Initialize an empty list to store the combined result
    combined = []
    
    # Iterate through the range of the maximum length
    for i in range(max_len):
        # Append the character from word1 if it exists, otherwise append nothing
        if i < len(word1):
            combined.append(word1[i])
        # Append the character from word2 if it exists, otherwise append nothing
        if i < len(word2):
            combined.append(word2[i])
    
    # Join the combined list with newlines for vertical concatenation
    result = "\n"
    result += "\n".join(combined)
    
    return result

def concat_words_vertically(word1, word2):
    # Concatenate the words directly with a newline between each letter
    combined = list(word1 + word2)
    
    # Join the letters with newlines for vertical concatenation
    result = "\n"
    result += "\n".join(combined)
    
    return result

def concat_words_cross(word2, word1):
    result = "\n"
    # Ensure word1 is centered in the cross
    mid_index = len(word1) // 2
    
    # Construct the vertical part of the cross
    vertical = [' ' * mid_index + c + ' ' * mid_index for c in word2]

    # Construct the horizontal part of the cross
    horizontal = word1
    
    # Combine the vertical and horizontal to form the cross
    for i in range(len(vertical)):
        if i == (len(vertical) // 2):
            result += f"{horizontal}\n"
        result += f"{vertical[i]}"
        if i < len(vertical) - 1:
            result += "\n"
    return result

def concat_words_x(word1, word2):
    # Determine the size of the grid based on the longer word
    max_len = max(len(word1), len(word2))
    
    # Initialize a grid with spaces
    grid = [[' ' for _ in range(max_len)] for _ in range(max_len)]

    # Place word2 ("BOMB") on the top-right to bottom-left diagonal
    for i in range(len(word2)):
        grid[i][max_len - 1 - i] = word2[i]
    
    # Place word1 ("APPLE") on the top-left to bottom-right diagonal
    for i in range(len(word1)):
        grid[i][i] = word1[i]

    # Combine the grid into a single string
    result = "\n"
    result += "\n".join("".join(row) for row in grid)
    
    return result

def float2string(value, round_decimal=2):
    formatted_string = f"{int(value) if value == 0 else round(value, round_decimal)}"
    return formatted_string

if __name__ == "__main__":
    print(concat_words_x("bomb", "apple"))
