import argparse
import json
import os
import os.path
import pandas as pd

from PIL import Image


def extract_bboxes_from_img(save_dir, dir_from, filename, boxes):
    created_bboxes = []

    for _, row in boxes.iterrows():
        x = row['bbox'][0]
        y = row['bbox'][1]
        width = row['bbox'][2]
        height = row['bbox'][3]

        im_path = os.path.join(dir_from, filename)

        im = Image.open(im_path)
        cropped = im.crop((x, y, x + width, y + height))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # new name: {old_image_name}__{image_id}__{bounding_box_id}.png
        cropped.save(os.path.join(
            save_dir, f"{filename[:-4]}__{row['image_id']}__{row['id']}.png"))

        new_im_info = {'im_id': row['image_id'], 'bbox_id': row['id'],
                       'width': cropped.size[0], 'height': cropped.size[1],
                       'category_id': row['category_id']}

        created_bboxes.append(new_im_info)

    return created_bboxes


def extract_bboxes(save_dir, dir_from, images, annotations):
    all_created_bboxes = []

    for _, row in images.iterrows():
        boxes = annotations[annotations['image_id'] == row['id']]

        created_bboxes = extract_bboxes_from_img(
            save_dir, dir_from, row['file_name'], boxes)
        all_created_bboxes.extend(created_bboxes)

    pd.DataFrame(all_created_bboxes).to_csv(
        os.path.join(save_dir, 'bboxes_data.csv'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', help='Folder where cropped images will be stored',
                        required=True, type=str)
    parser.add_argument('--dir-from', help='Folder where original images are stored',
                        required=True, type=str)
    parser.add_argument('--data-json', help='Path to file data describing how to crop images',
                        required=True, type=str)

    args = parser.parse_args()

    with open(args.data_json, 'r') as f:
        data = json.load(f)

    images = pd.DataFrame(data['images'])
    annotations = pd.DataFrame(data['annotations'])

    extract_bboxes(args.save_dir, args.dir_from, images, annotations)


if __name__ == '__main__':
    main()
