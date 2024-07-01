import cv2
import os
import face_recognition
import numpy as np
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import pyttsx3

def load_known_faces(dataset_path):
    known_faces = []
    known_labels = []

    for filename in os.listdir(dataset_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path)
            
            if img is not None:
                # Convert to RGB for face_recognition
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                 
                # Encode the face
                face_encoding = face_recognition.face_encodings(rgb_img)
                
                if len(face_encoding) > 0:
                    known_faces.append(face_encoding[0])
                    label = os.path.splitext(filename)[0]
                    known_labels.append(label)

    return known_faces, known_labels

def send_email(image_path):
    # Set up your email configuration
    sender_email = "abc@gmail.com"
    receiver_email = "xyz@gmail.com"
    password = "sender gmail app password"

    # Create the email content
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Unauthorized Access Detected!"

    # Add text to the email
    body = "Unauthorized access detected. Image of the intruder is attached."
    msg.attach(MIMEText(body, 'plain'))

    # Attach the captured image
    with open(image_path, 'rb') as file:
        img_data = MIMEImage(file.read(), name=os.path.basename(image_path))
        msg.attach(img_data)

    # Connect to the SMTP server and send the email
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    # Initialize the classifier
    cascPath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Specify the path to the dataset folder
    dataset_path = r"C:\Users\PMLS\Desktop\facial recognition model\dataset"

    # Load known faces from the dataset
    known_faces, known_labels = load_known_faces(dataset_path)

    # Open a connection to the webcam (usually 0 or 1, depending on your setup)
    cap = cv2.VideoCapture(0)

    # Variables for tracking unknown face duration
    start_time = None
    unknown_duration_threshold = 4  # in seconds

    # Variable to track if known face is detected
    known_face_detected = False

    while True:
        # Capture video frame-by-frame
        ret, img = cap.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Check if any known faces are detected
        known_face_detected = False

        # Iterate through detected faces in the frame
        for (x, y, w, h) in faces:
            face = img[y:y + h, x:x + w]
            
            # Convert to RGB for face_recognition
            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
            # Encode the face
            face_encoding = face_recognition.face_encodings(rgb_face)
            
            if len(face_encoding) > 0:
                face_encoding = face_encoding[0]
                matches = face_recognition.compare_faces(known_faces, face_encoding)
                distances = face_recognition.face_distance(known_faces, face_encoding)

                # Find the index of the lowest distance
                best_match_index = np.argmin(distances)

                # If there is a match, display the name of the person on the frame
                if matches[best_match_index]:
                    known_face_detected = True
                    name = known_labels[best_match_index]
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # If at least one known face is detected, grant access and send an email
        if known_face_detected:
            start_time = None  # Reset the timer
            text_to_speech("Access Granted")
            send_email("known_face.jpg")  # Send an email with the image
            time.sleep(10)
            
        # If no known face is detected, start or reset the timer
        else:
            if start_time is None:
                start_time = time.time()
            else:
                elapsed_time = time.time() - start_time
                if elapsed_time >= unknown_duration_threshold:
                    # Trigger actions for unauthorized access (e.g., play sound, send email)
                    print("Unauthorized access detected!")
                    cv2.imwrite("intruder.jpg", img)  # Save the image of the intruder
                    send_email("intruder.jpg")  # Send an email with the image
                    start_time = None  # Reset the timer

        # Display the resulting frame
        cv2.imshow('Face recognition', img)

        # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
