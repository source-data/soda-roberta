'''
Application-wide preferences.
'''


class Config:
    dirignore = ['__MACOSX', '.DS_Store']
    vocab_size = 54_000
    max_length = 514  # tokens!
    min_char_length = 120  # characters
    split_ratio = {
        'train': 0.7,
        'eval': 0.2,
        'test': 0.1,
        'max_eval': 10_000,
        'max_test': 10_000,
    }
    outside_of_entity_label = "O"  # label used for token, chars outside of a named entity


config = Config()
