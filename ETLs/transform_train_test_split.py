import yaml
import random


from pathlib import Path


def train_test_split(path_img:Path, path_labels:Path, train_size:float) -> None:
    '''Splits the annotated data into train and test sets

    The inputs are comming out of label-studio annotated data in the YOLO format.
    This method will split and organize the data to be used by YOLO.V8. The name
    of the test set is called val.

    Args:
        path_img (Path): path to labeled images
        path_labels (Path): path to the images labels, bounding boxes
        train_size (float): how to split the data

    Returns:
        None
    '''
    
    imgs = [file for file in path_img.rglob('*') if file.is_file()]
    labels = [label for label in path_labels.rglob('*.txt') if label.is_file()]
    
    # sorting the lists
    imgs.sort(), labels.sort()

    # semi-randomly selecting train data
    random_train_index = random.sample(
        range(len(imgs)), round(len(imgs)*train_size)
    )
    img_train = [imgs[index] for index in random_train_index]
    label_train = [labels[index] for index in random_train_index]

    # selecting test data
    img_test = [img for img in imgs if img not in img_train]
    label_test = [label for label in labels if label not in label_train]

    train = path_img.parent / 'train'
    val = path_img.parent / 'val'
    
    for folder in [train, val]:
        for sub in ['images', 'labels']:
            full_path = folder / sub
            if not full_path.exists():
                full_path.mkdir(parents=True)

    [img.rename(train / 'images' / img.name) for img in img_train]
    [img.rename(val / 'images' / img.name) for img in img_test]
    [label.rename(train / 'labels' / label.name) for label in label_train]
    [label.rename(val / 'labels' / label.name) for label in label_test]

    for folder in [path_img, path_labels]:
        if folder.exists() and folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()

    with open(path_img.parent / 'classes.txt', 'r') as f:
        classes = f.read().splitlines()

    data = {
        'path'  : str(path_img.parent),
        'train' : 'train/images',
        'val'   : 'val/images',
        'names': {index:val for index, val in enumerate(classes)},
    }

    with open(path_img.parent / 'data.yaml', 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    return


def main():
    annoted_path = Path(
        'sample_data/anotated/project-5-at-2024-02-24-19-48-ced05f33'
    )
    path_img = annoted_path / 'images'
    path_labels = annoted_path / 'labels'

    train_test_split(
        path_img=path_img,
        path_labels=path_labels,
        train_size=0.8
    )


if __name__=='__main__':
    main()
