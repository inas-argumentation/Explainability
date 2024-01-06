import numpy as np

# Split text into multiply smaller parts to make them suitable to be processed by BERT.
def split_sample(tokenizer, text, additionally_extend_last_split=True):
    sample_tokenized = tokenizer.tokenize(text)
    num_splits = int(np.ceil(len(sample_tokenized) / 400))  # Number of splits the text is divided into

    # Assign split index to each word
    words = []
    current_word = []
    split_counts = {i: 0 for i in range(0, num_splits)}  # Counts number of tokens per split
    current_split = 0
    tokens_per_split = 500
    for i in range(len(sample_tokenized)):
        token = sample_tokenized[i]

        if token[:2] != "##":  # A new word starts
            if len(current_word) > 0:
                words.append({"tokens": current_word, "num_tokens": len(current_word), "word": "".join(current_word),
                              "splits": [current_split]})
                split_counts[current_split] += len(current_word)
                if split_counts[current_split] >= tokens_per_split:
                    current_split += 1
                    tokens_per_split = 400
            current_word = [token]
        else:  # The current token is a continuation of the current word
            current_word.append(token[2:])
    words.append({"tokens": current_word, "num_tokens": len(current_word), "word": "".join(current_word),
                  "splits": [current_split]})
    split_counts[current_split] += len(current_word)

    # If the last split is very short, merge it with the second to last one
    if split_counts[num_splits - 1] <= 100 and num_splits > 1 and split_counts[num_splits - 2] + split_counts[
        num_splits - 1] <= 510:
        for word in words:
            word["splits"] = word["splits"] if num_splits - 1 not in word["splits"] else [num_splits - 2]
        num_splits = num_splits - 1
        split_counts[num_splits - 1] += split_counts[num_splits]
        del split_counts[num_splits]

    # Make splits overlap by 100 tokens by extending each split 100 tokens backward.
    # If additionally_extend_last_split == True, the last split is extended backwards to consist of about 480 tokens.
    add_counter = 100
    current_split = num_splits
    for i in range(len(words)):
        current_word = words[-(i + 1)]
        if (add_counter < 100 or (current_split in split_counts and split_counts[current_split] < 480 and additionally_extend_last_split)) \
                and current_split not in current_word["splits"] and split_counts[current_split] + current_word[
            "num_tokens"] <= 510:
            current_word["splits"].append(current_split)
            add_counter += current_word["num_tokens"]
            split_counts[current_split] += current_word["num_tokens"]
        elif add_counter >= 100 or split_counts[current_split] + current_word["num_tokens"] > 510:
            current_split = current_word["splits"][0]
            add_counter = 0

    return words, num_splits, split_counts