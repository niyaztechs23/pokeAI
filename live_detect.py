import cv2
import mss
import numpy as np
import torch
import torch.nn as nn
import json
import time
from PIL import Image
from torchvision import models, transforms

# -----------------------------
# Load class names
# -----------------------------
with open("class_names.json") as f:
    class_names = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load model
# -----------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("pokemon_classifier.pt", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# Image transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Screen capture region
# -----------------------------
monitor = {
    "top": 664,
    "left": 450,
    "width": 492,
    "height": 314
}

sct = mss.mss()

last_prediction = ""
last_confidence = 0
last_time = 0

print("Press Q to quit.")

while True:

    screenshot = sct.grab(monitor)

    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Predict every 0.5 sec
    if time.time() - last_time > 0.5:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]

        conf, idx = torch.max(probs, 0)

        last_prediction = class_names[idx.item()]
        last_confidence = conf.item() * 100
        last_time = time.time()

    cv2.putText(
        frame,
        f"{last_prediction} ({last_confidence:.1f}%)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("AMGLITCH PokeAI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()