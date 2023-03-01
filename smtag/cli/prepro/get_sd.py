from argparse import ArgumentParser
from ...smartnode import Collection


if __name__ == "__main__":
    parser = ArgumentParser(description="Download the SourceData xml tagged dataset.")
    parser.add_argument("--name", default="PUBLICSEARCH", help="The name of the collection to download.")
    parser.add_argument("--dest_dir", help="The destination dir to save the xml files.")
    parser.add_argument("--api", default="sdapi", choices=["sdapi","neo"], help="Data source")

    args = parser.parse_args()
    collection_name = args.name
    dest_dir = args.dest_dir

    if args.api == "sdapi":
        c = Collection(auto_save=True, sub_dir=dest_dir).from_sd_REST_API(collection_name)
    if args.api == "neo":
        c = Collection(auto_save=True, sub_dir=dest_dir).from_neo(collection_name)
    print(f"downloaded collection: {c.props}")