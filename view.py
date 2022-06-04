import fiftyone as fo
import argparse

def viz(data_path: str, labels_path: str) -> None:
    dataset = fo.Dataset.from_dir(
        data_path=data_path,
        labels_path=labels_path,
        dataset_type=fo.types.COCODetectionDataset,
    )

    sess = fo.launch_app(dataset)
    sess.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', help='Folder with images',
                        required=True, type=str)
    parser.add_argument('--labels-path', help='JSON file with annotations',
                        required=True, type=str)

    args = parser.parse_args()

    viz(args.data_path, args.labels_path)