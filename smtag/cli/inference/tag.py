import argparse
from ast import parse
from ...pipeline import SmartTagger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartTagging of free text.")
    parser.add_argument("text", nargs="?", default="We studied mice with genetic ablation of the ERK1 gene in brain and muscle.", help="The text to tag.")
    parser.add_argument("--local_model_dir", help="For test, the local directory where the models can be found")
    args = parser.parse_args()
    text = args.text
    local_model_dir = args.local_model_dir
    if local_model_dir is not None:
        smtg = SmartTagger(
            panelizer_source=local_model_dir+"/PANELIZATION",
            ner_source=local_model_dir+"/NER",
            geneprod_roles_source=local_model_dir+"/GENEPROD_ROLES",
            small_mol_roles_source=local_model_dir+"/SMALL_MOL_ROLES"
        )
    else:
        smtg = SmartTagger()
    tagged = smtg(text)
    print(tagged)
