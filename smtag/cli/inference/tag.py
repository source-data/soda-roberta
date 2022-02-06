import argparse
from ...pipeline import SmartTagger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartTagging of free text.")
    parser.add_argument("text", nargs="?", default="We studied mice with genetic ablation of the ERK1 gene in brain and muscle.", help="The text to tag.")
    args = parser.parse_args()
    text = args.text
    smtg = SmartTagger()
    tagged = smtg(text)
    print(tagged)
