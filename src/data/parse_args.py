from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Module for preparing data from a data in the format ShareGPT.")
    parser.add_argument("-d", "--dataset",
                        help="The local or remote path to the huggingface dataset.",
                        type=str,
                        default="Open-Orca/SlimOrca",
                        required=False)
    parser.add_argument("-t", "--tokenizer",
                        help="The local or remote path to the huggingface tokenizer.",
                        type=str,
                        default="microsoft/phi-2",
                        required=False)
    parser.add_argument("-b", "--batch_size",
                        help="The batch for pre-processing the dataset on the CPUs.",
                        type=int,
                        default=1,
                        required=False)
    parser.add_argument("-c", "--target_column",
                        help="The containing the column to process.",
                        type=str,
                        default="conversations",
                        required=False)
    parser.add_argument("-l", "--max_length",
                        help="The maximum length of the tokenized text.",
                        type=int,
                        default=1024,
                        required=False)
    parser.add_argument("-s", "--subset",
                        help="The maximum length of the tokenized text.",
                        type=int,
                        default=13_000_000,
                        required=False)
    parser.add_argument("-p", "--num_proc",
                        help="The number of parallel processes.",
                        type=int,
                        default=64,
                        required=False)
    parser.add_argument("-o", "--output-folder",
                        help="Where the output folder will be stored.",
                        type=str,
                        required=True)
    
    return parser.parse_args()