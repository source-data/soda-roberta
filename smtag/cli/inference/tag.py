import argparse
from ast import parse
from ...pipeline_full_caption import SmartTagger as SmartTaggerFull
from ...pipeline import SmartTagger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartTagging of free text.")
    parser.add_argument("text", nargs="?", default="We studied mice with genetic ablation of the ERK1 gene in brain and muscle.", help="The text to tag.")
    parser.add_argument("--local_model_dir", help="For test, the local directory where the models can be found")
    parser.add_argument("-P", "--tag_panels", action="store_true", help="If activated, will do the NER and ROLES in full caption instead of panel by panel.")
    args = parser.parse_args()
    text = args.text
    local_model_dir = args.local_model_dir

    if args.tag_panels:
        if local_model_dir is not None:
            smtg = SmartTagger(
                panelizer_source=local_model_dir+"/PANELIZATION",
                ner_source=local_model_dir+"/NER",
                geneprod_roles_source=local_model_dir+"/GENEPROD_ROLES",
                small_mol_roles_source=local_model_dir+"/SMALL_MOL_ROLES",
            )
        else:
            smtg = SmartTagger(
                panelizer_source="EMBO/sd-panelization-v2",
                ner_source="EMBO/sd-ner-v2",
                geneprod_roles_source="EMBO/sd-geneprod-roles-v2",
                small_mol_roles_source="EMBO/sd-smallmol-roles-v2",

                )
    else:
        if local_model_dir is not None:
            smtg = SmartTaggerFull(
                panelizer_source=local_model_dir+"/PANELIZATION",
                ner_source=local_model_dir+"/NER",
                geneprod_roles_source=local_model_dir+"/GENEPROD_ROLES",
                small_mol_roles_source=local_model_dir+"/SMALL_MOL_ROLES",
            )
        else:
            smtg = SmartTaggerFull(
                panelizer_source="EMBO/sd-panelization-v2",
                ner_source="EMBO/sd-ner-v2",
                geneprod_roles_source="EMBO/sd-geneprod-roles-v2",
                small_mol_roles_source="EMBO/sd-smallmol-roles-v2",
            )

    tagged = smtg(text)
    print(tagged)
