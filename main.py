from torchvision import transforms
import torch
import onnx
import numpy as np
from multiprocessing import Process, Queue
from fastai.vision import normalize, imagenet_stats
import cv2
import caffe2.python.onnx.backend as backend


def preprocess_img(img):
    """
    Prepare image to be used by caffe2 model
    """
    # Resize the face so it is 64x64
    img = cv2.resize(img, (64, 64))

    # Change shape form HWC to CHW
    img = np.rollaxis(img, 2, 0)

    # Normailse image
    mn,std = imagenet_stats
    img = img / 255
    img = normalize(torch.FloatTensor(img),torch.FloatTensor(mn),torch.FloatTensor(std))

    # convert back to numpy array
    img = img.numpy()

    # return img ready for batching 
    return np.expand_dims(img, axis=0)


def softmax(batch):
    """
    Compute probability values for each sets of scores in .
    """
    e_batch = np.exp(batch - np.max(batch,axis=1,keepdims=True))
    return e_batch / e_batch.sum(axis=1,keepdims=True)


def predict(model, input_batch):
    """
    Return probabilities outpout by the model
    """
    # Retrieve log probs from the model
    outputs = model.run(input_batch)

    # Calculate and return probabilities
    return softmax(outputs[0])


def classify_frame(model, inputQueue, outputQueue):
    """
    Uses multiprocessing library to return predictions.
    """
    while True:
        if not inputQueue.empty():
            # Retrieve batch form input queue
            model_input = inputQueue.get()

            # Get probabilities
            predictions = predict(caffe_resnet18, model_input)

            # Put probabilities into the output queue
            outputQueue.put(predictions)


def main():
    # Start the capture from the webcam
    cap = cv2.VideoCapture(0)

    # Load the haarcascade to find faces in the frames
    # main pc
    face_cascade = cv2.CascadeClassifier('/home/matt/.conda/pkgs/opencv-3.4.1-py36_blas_openblash829a850_201/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    # laptop
    # face_cascade = cv2.CascadeClassifier('/home/matt/Developer/anaconda3/pkgs/opencv-3.4.3-py35_blas_openblash829a850_200/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

    # Set max input and output queue size to 1
    inputQueue = Queue(maxsize=1)
    outputQueue = Queue(maxsize=1)

    # Start the multiprocessing
    p = Process(target=classify_frame, args=(caffe_resnet18, inputQueue, outputQueue))
    p.daemon = True
    p.start()

    # Declare empty predictions list
    predictions = []

    # Main loop
    # Each loop processes one frame
    print("[INFO] Starting capture...")
    while True:
        # Set the batch to have a random array to define the shape
        model_input = np.random.rand(1, 3, 64, 64).astype(np.float32)

        # Get the frame from the webcam object
        ret, frame = cap.read()

        # Convert frame to gray for the haarcascade
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get a list of all faces found in the image
        faces = face_cascade.detectMultiScale(grey, 1.3, 7)

        # Loop through each face
        for (x, y, w, h) in faces:
            # Crop the face from the frame
            face = frame[y:y+h, x:x+w]

            # Process the face and add it to the batch
            model_input = np.concatenate((model_input, preprocess_img(face)), axis=0)

        # If input queue is empty give it the batch
        if inputQueue.empty():
            inputQueue.put(model_input)

        # If output queue is not empty get the predictions
        if not outputQueue.empty():
            predictions = outputQueue.get()

        # Loop through each face and prediction
        for (x, y, w, h), prediction in zip(faces, predictions[1:]):
            # Define text to be shown
            if prediction[0] >= 0.5:
                text = "Happy :)  {:.1f}%".format(prediction[0] * 100)
            else: 
                text = "Not happy  {:.1f}%".format(prediction[1] * 100)

            # Draw rectangle around face and put the label on
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.rectangle(frame, (x, y - 30), (x + 205, y), (0, 200, 0), cv2.FILLED)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Open the window
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release cap from memory and destroy window
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':

    # Load the model into memory
    print("[INFO] Loading model into memory...")
    onnx_model = onnx.load('./onnx_models/happy.onnx')
    caffe_resnet18 = backend.prepare(onnx_model, device='CPU')

    # Run main loop
    main()