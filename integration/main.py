import sys
import cv2
import torch


# ---------------------------
# Add Paths
# ---------------------------
sys.path.append("../cv_module")
sys.path.append("../ml_module")
sys.path.append("../rag_module")


# ---------------------------
# Import CV
# ---------------------------
from detect_objects_v2 import (
    preprocess,
    color_segmentation,
    edge_detection,
    clean_mask,
    find_objects
)


# ---------------------------
# Import ML
# ---------------------------
from torchvision import models, transforms
import torch.nn as nn


# ---------------------------
# Import RAG
# ---------------------------
from rag_system import retrieve, generate_answer


# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Load Classifier
# ---------------------------
def load_classifier():

    model = models.mobilenet_v2(weights="DEFAULT")

    for p in model.parameters():
        p.requires_grad = False

    model.classifier[1] = nn.Linear(model.last_channel, 3)

    model.load_state_dict(
        torch.load(
            "../ml_module/warehouse_classifier.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model


classifier = load_classifier()


# ---------------------------
# Transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


labels = ["heavy", "fragile", "hazardous"]


# ---------------------------
# Classify
# ---------------------------
def classify_object(crop):

    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    img = transform(
        transforms.ToPILImage()(img)
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        out = classifier(img)

    return labels[out.argmax(1).item()]


# ---------------------------
# Main Pipeline
# ---------------------------
def main():

    print("\nRunning Full Warehouse AI Pipeline\n")

    img = cv2.imread("../cv_module/test3.jpg")

    if img is None:
        print("Image not found")
        return


    original = img.copy()


    # -------- CV --------
    prep = preprocess(img)
    mask = color_segmentation(prep)
    edges = edge_detection(mask)
    cleaned = clean_mask(edges)

    centers = find_objects(cleaned, original)

    print("\nDetected Objects:", len(centers))

    if len(centers) == 0:
        return


    # -------- ML + RAG --------
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    obj_id = 1


    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 2000:
            continue


        x, y, w, h = cv2.boundingRect(cnt)

        crop = img[y:y+h, x:x+w]

        if crop.size == 0:
            continue


        # ML classification
        category = classify_object(crop)

        print(f"\nObject {obj_id}: {category}")


        # RAG automatic explanation
        query = f"How should {category} items be handled?"
        contexts = retrieve(query)
        answer = generate_answer(query, contexts)

        print("Handling Instructions:")
        print(answer)
        print("-" * 40)


        obj_id += 1


    cv2.imshow("Detected Objects", original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # -------- Optional Q&A --------
    print("\nYou may now ask additional warehouse questions.")
    print("Type 'exit' to finish.\n")

    while True:

        user_query = input("Ask: ")

        if user_query.lower() == "exit":
            break

        contexts = retrieve(user_query)
        answer = generate_answer(user_query, contexts)

        print("\nAnswer:")
        print(answer)
        print("-" * 40)


# ---------------------------
if __name__ == "__main__":
    main()
