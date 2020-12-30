'''
Application-wide preferences.
'''


class Config:
    dirignore = ['__MACOSX', '.DS_Store']
    vocab_size = 54_000
    max_length = 512  # token
    min_char_length = 120  # characters
    split_ratio = {
        'train': 0.7,
        'eval': 0.2,
        'test': 0.1,
    }


config = Config()
