import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

class EmotionDetector:
    def __init__(self, model_path):
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        
        # Get the actual number of classes from the model
        self.num_classes = self.model.output_shape[-1]
        print(f"Model has {self.num_classes} output classes")
        
        # Define class names for 7 classes
        self.class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        print("Class names:", self.class_names)
        
    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to 48x48 (model input size)
        image = cv2.resize(image, (48, 48))
        
        # Normalize pixel values
        image = image / 255.0
        
        # Add batch dimension and channel dimension
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = np.expand_dims(image, axis=-1) # Add channel dimension
        
        return image
    
    def predict_emotion(self, image):
        """Predict emotion from image"""
        processed_image = self.preprocess_image(image)
        predictions = self.model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        
        return self.class_names[predicted_class], confidence
    
    def detect_from_webcam(self):
        """Real-time emotion detection from webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Press 'q' to quit webcam")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Make a copy for display
            display_frame = frame.copy()
            
            # Predict emotion
            emotion, confidence = self.predict_emotion(frame)
            
            # Display emotion on frame
            cv2.putText(display_frame, f"Emotion: {emotion}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Confidence: {confidence:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Emotion Detection', display_frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def detect_from_image(self, image_path):
        """Detect emotion from image file"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        emotion, confidence = self.predict_emotion(image)
        
        # Display result
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Predicted Emotion: {emotion} (Confidence: {confidence:.2f})")
        plt.axis('off')
        plt.show()
        
        return emotion, confidence

# Main execution
if __name__ == "__main__":
    # Initialize detector with your model path
    detector = EmotionDetector('best_emotion_detection.h5')
    
    print("Emotion Detection System")
    print("1. Webcam detection")
    print("2. Image file detection")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == '1':
        detector.detect_from_webcam()
    elif choice == '2':
        image_path = input("Enter image path: ")
        detector.detect_from_image(image_path)
    else:
        print("Invalid choice")