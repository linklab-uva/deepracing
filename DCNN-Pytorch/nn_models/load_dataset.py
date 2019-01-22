import argparse
import data_loading.data_loaders as loaders
def main():
    parser = argparse.ArgumentParser(description="DatasetLoader")
    parser.add_argument("--dataset_file", type=str, required=True, help="Labels file to use")
    parser.add_argument("--dataset_type", type=str, required=True, help="Type of dataset to load")
    parser.add_argument("--context_length", type=int, default=10, help="Context length for sequence datasets")
    parser.add_argument("--sequence_length", type=int, default=1, help="Sequence length for sequence datasets")
    args = parser.parse_args()
    size=(66,200)
    fp = args.dataset_file
    dataset_type = args.dataset_type
    context_length = args.context_length
    sequence_length = args.sequence_length
    if dataset_type=='optical_flow':
        ds = loaders.F1OpticalFlowDataset(fp, size, context_length = context_length, sequence_length = sequence_length)
    elif dataset_type=='raw_images':
        ds = loaders.F1ImageSequenceDataset(fp, size, context_length = context_length, sequence_length = sequence_length)
    elif dataset_type=='combined':
        ds = loaders.F1CombinedDataset(fp, size, context_length = context_length, sequence_length = sequence_length)
    elif dataset_type=='static_images':
        ds = loaders.F1ImageDataset(fp, size)
    else:
        raise NotImplementedError('Dataset of type: ' + dataset_type + ' not implemented.')
    ds.loadFiles()
    ds.writePickles()

if __name__ == '__main__':
    main()