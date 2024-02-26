from ultralytics import YOLO


# input the path to your pretrainned yolo model according to your needs
model = YOLO('<path_to_your_local_yolo_model>/yolov8s.pt')


def main():
    model.train(
        # use the path to your data such as:
        # sample_data/annotated/project-5-at-2024-02-24-19-48-ced05f33/data.yaml
        data='<path_to_your_local_data_yaml_file>/data.yaml',
        epochs=100, 
        imgsz=640,
        workers=8,
        pretrained=True,
        name='your_custom_model_name'
    )


if __name__=='__main__':
    main()    
