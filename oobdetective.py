import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter

def recognize_objects(frame):
    """
    Function to identify objects/people using the webcam
    """
    # Preprocess the image to fit the model
    resized_frame = cv2.resize(frame, tuple(input_shape))
    input_data = np.expand_dims(resized_frame, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Model predictions
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Ensure consistent dimensions
    min_length = min(len(boxes), len(classes), len(scores))
    boxes = boxes[:min_length]
    classes = classes[:min_length]
    scores = scores[:min_length]

    # Filter for predictions with a confidence threshold and convert numeric classes to object names
    threshold = 0.50  # %
    detections = [(box, labels[int(cls)], score) for box, cls, score in zip(boxes, classes, scores) if score > threshold and int(cls) < len(labels)]

    # Draw predictions on the image
    for box, obj_name, score in detections:
        ymin, xmin, ymax, xmax = box
        xmin = int(xmin * frame.shape[1])
        xmax = int(xmax * frame.shape[1])
        ymin = int(ymin * frame.shape[0])
        ymax = int(ymax * frame.shape[0])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = f"{obj_name}: {score:.2f}"
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

if __name__ == '__main__':

    # TensorFlow Lite model for object recognition
    model_path = "/home/christian/Scrivania/PycharmProjects/progettoDetective/tflite"
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape'][1:3]

    # Configuration file containing labels (classes)
    label_path = "/home/christian/Scrivania/PycharmProjects/progettoDetective/coco_labels.txt"
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Frame from the webcam
        ret, frame = cap.read()
        frame = cv2.resize(frame, (400, 400))

        # Object detection in the frame
        frame_with_objects = recognize_objects(frame)

        # Show frame with objects
        cv2.imshow('Object Recognition', frame_with_objects)

        # Delay and exit check if user presses 'q'
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    # Release webcam and close window
    cap.release()
    cv2.destroyAllWindows()












