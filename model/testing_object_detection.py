import numpy as np
import supervision as sv


from ultralytics import YOLO


VIDEO_PATH = "<path_to_your_local_test_video>"
model = YOLO("<path_to_your_local_trainned_model>")


def process_frame(frame: np.ndarray) -> np.ndarray:
    '''Process a frame of an input video or image and draws the boxes around the
    detected objects.

    This function takes an input frame and performs object detection using a 
    pre-trained model. It annotates the detected objects with bounding boxes and
    labels, and returns the annotated frame.

    Args:
        frame (np.ndarray): The input frame (image) to be processed.

    Returns:
        np.ndarray: The processed frame with annotated bounding boxes and labels.
    '''

    results = model(frame, imgsz=1280)[0]
    detections = sv.Detections.from_ultralytics(results)
    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
    labels = [
        f"{model.names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _ in detections
    ]
    frame = box_annotator.annotate(
        scene=frame, detections=detections, labels=labels
    )
    return frame


def main():
    sv.process_video(
        source_path=VIDEO_PATH,
        target_path="<path_to_save_the_test_video>",
        callback=process_frame
    )


if __name__=="__main__":
    main()
