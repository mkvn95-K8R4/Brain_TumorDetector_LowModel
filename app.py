from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model("brain_tumor_model.h5")
classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

tumor_details = {
    "Glioma": {
        "description": "Glioma is a type of tumor that occurs in the brain and spinal cord, originating from glial cells.",
        "causes": [
            "Genetic mutations",
            "Exposure to high levels of radiation",
            "Family history (rare cases)"
        ],
        "symptoms": [
            "Persistent headaches",
            "Seizures",
            "Difficulty in speaking",
            "Vision problems"
        ],
        "suggestions": [
            "Consult a neurologist immediately",
            "Avoid self-medication"
        ]
    },

    "Meningioma": {
        "description": "Meningioma arises from the meninges, the membranes that surround the brain and spinal cord.",
        "causes": [
            "Radiation exposure",
            "Hormonal factors",
            "Genetic conditions"
        ],
        "symptoms": [
            "Headaches",
            "Blurred or double vision",
            "Memory problems",
            "Hearing loss"
        ],
        "suggestions": [
            "Consult a neurosurgeon",
            "Regular monitoring if small",
        ]
    },

    "Pituitary": {
        "description": "Pituitary tumors occur in the pituitary gland and may affect hormone production.",
        "causes": [
            "Unknown exact cause",
            "Genetic mutations",
            "Hormonal imbalances"
        ],
        "symptoms": [
            "Vision problems",
            "Hormonal changes",
            "Fatigue",
            "Unexplained weight changes"
        ],
        "suggestions": [
            "Consult an endocrinologist",
            "Hormone level tests",  
        ]
    }
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if not file: return "No file"
    
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # 1. Prediction logic
    img_array = cv2.imread(filepath)
    img_resized = cv2.resize(img_array, (224, 224))
    img_norm = img_resized / 255.0
    img_reshape = np.reshape(img_norm, (1, 224, 224, 3))
    prediction = model.predict(img_reshape)
    result = classes[np.argmax(prediction)]

    # 2. Advanced Internal Highlighting
    boxed = cv2.resize(img_array, (224, 224))
    
    if result != "No Tumor":
        gray = cv2.cvtColor(boxed, cv2.COLOR_BGR2GRAY)
        
        # A. Blur to smooth out brain texture
        blurred = cv2.GaussianBlur(gray, (13, 13), 0)
        
        # B. Use Canny to find the 'walls' of the tumor
        edged = cv2.Canny(blurred, 30, 80)
        
        # C. CIRCULAR MASK: This is the "Skull Stripper"
        # We create a black image and draw a white circle in the middle
        mask = np.zeros(edged.shape, dtype=np.uint8)
        center = (edged.shape[1] // 2, edged.shape[0] // 2)
        radius = int(min(edged.shape) * 0.38) # Target the inner 76% of the scan
        cv2.circle(mask, center, radius, 255, -1)
        
        # Apply the mask: Everything outside the circle becomes black (no edges)
        masked_edges = cv2.bitwise_and(edged, mask)
        
        # D. Dilate to make the tumor boundary a solid shape
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(masked_edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Pick the largest object found INSIDE the brain circle
            largest_cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_cnt) > 150: # Ignore tiny specks
                x, y, w, h = cv2.boundingRect(largest_cnt)
                cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # 3. Save result
    box_path = os.path.join("static", "boxed_" + file.filename)
    cv2.imwrite(box_path, boxed)

    details = tumor_details.get(result, {})

    return render_template(
        "result.html",
        result=result,
        image_path=filepath,
        box_path=box_path,
        details=details
    )

if __name__ == '__main__':
    if not os.path.exists('static'): os.makedirs('static')
    app.run(debug=True)