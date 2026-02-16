import cv2
import numpy as np
from scipy.spatial import distance as dist


# -------------------------------
# Simple Centroid Tracker
# -------------------------------
class CentroidTracker:

    def __init__(self, max_dist=60):

        self.nextID = 1
        self.objects = {}
        self.max_dist = max_dist


    def update(self, centers):

        if len(centers) == 0:
            return self.objects


        if len(self.objects) == 0:

            for c in centers:
                self.objects[self.nextID] = c
                self.nextID += 1

            return self.objects


        objectIDs = list(self.objects.keys())
        objectCenters = list(self.objects.values())


        D = dist.cdist(
            np.array(objectCenters),
            np.array(centers)
        )


        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]


        usedRows = set()
        usedCols = set()


        for r, c in zip(rows, cols):

            if r in usedRows or c in usedCols:
                continue

            if D[r, c] > self.max_dist:
                continue

            objID = objectIDs[r]
            self.objects[objID] = centers[c]

            usedRows.add(r)
            usedCols.add(c)


        unusedCols = set(range(len(centers))) - usedCols

        for c in unusedCols:

            self.objects[self.nextID] = centers[c]
            self.nextID += 1


        return self.objects


# -------------------------------
# Preprocessing
# -------------------------------
def preprocess(img):

    img = cv2.resize(img, (900, 600))

    blur = cv2.GaussianBlur(img, (5, 5), 0)

    return blur


# -------------------------------
# Color Segmentation
# -------------------------------
def color_segmentation(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Cardboard box color range
    lower = np.array([5, 40, 40])
    upper = np.array([25, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    return mask


# -------------------------------
# Edge Detection
# -------------------------------
def edge_detection(mask):

    edges = cv2.Canny(mask, 50, 150)

    return edges


# -------------------------------
# Noise Cleaning
# -------------------------------
def clean_mask(edges):

    kernel = np.ones((5, 5), np.uint8)

    dilated = cv2.dilate(edges, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    return eroded


# -------------------------------
# Object Detection
# -------------------------------
def find_objects(cleaned, original):

    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    centers = []
    count = 0


    for cnt in contours:

        area = cv2.contourArea(cnt)

        # Remove noise
        if area < 2000:
            continue


        x, y, w, h = cv2.boundingRect(cnt)

        # Shape filter
        if w < 30 or h < 30:
            continue


        # Centroid using moments
        M = cv2.moments(cnt)

        if M["m00"] != 0:

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        else:

            cx = x + w // 2
            cy = y + h // 2


        centers.append((cx, cy))
        count += 1


        # Draw bounding box
        cv2.rectangle(
            original,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # Draw center
        cv2.circle(
            original,
            (cx, cy),
            5,
            (0, 0, 255),
            -1
        )


        label = f"{w}x{h}px"

        cv2.putText(
            original,
            label,
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )


        # Console log
        print(f"Object {count}")
        print(f" Center: ({cx}, {cy})")
        print(f" Size: {w} x {h}")
        print("-" * 25)


    print("Total Objects:", count)

    return centers


# -------------------------------
# Main
# -------------------------------
def main():

    image_path = "test1.jpg"

    tracker = CentroidTracker()


    img = cv2.imread(image_path)

    if img is None:
        print("Image not found")
        return


    original = img.copy()

    prep = preprocess(img)

    mask = color_segmentation(prep)

    edges = edge_detection(mask)

    cleaned = clean_mask(edges)

    centers = find_objects(cleaned, original)

    objects = tracker.update(centers)


    # Draw IDs
    for objID, center in objects.items():

        text = f"ID {objID}"

        cv2.putText(
            original,
            text,
            (center[0]-10, center[1]-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )


    cv2.imshow("Detection + IDs", original)
    cv2.imshow("Mask", mask)
    cv2.imshow("Edges", edges)
    cv2.imshow("Cleaned", cleaned)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------------------------
if __name__ == "__main__":

    main()
