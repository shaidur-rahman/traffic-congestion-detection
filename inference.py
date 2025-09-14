import cv2
import time
from ultralytics import YOLO
from collections import defaultdict

# Parameters
STATIONARY_THRESHOLD = 10   # seconds
IOU_THRESHOLD = 0.5         # for tracking consistency

def iou(box1, box2):
    """Calculate IoU (Intersection over Union) between two boxes"""
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2b - x1b) * (y2b - y1b)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def main():
    model = YOLO("runs/detect/illegal_parking_yolo/weights/best.pt")

    cap = cv2.VideoCapture("test_videos/dhaka_test.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("outputs/result.mp4", fourcc, 30.0,
                          (int(cap.get(3)), int(cap.get(4))))

    # Track vehicles {id: {last_box, first_seen_time, stationary_flag}}
    vehicles = defaultdict(dict)
    next_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        current_time = time.time()

        for box in results.boxes.xyxy:  # bounding box
            x1, y1, x2, y2 = box.tolist()
            detected_box = (x1, y1, x2, y2)

            # Try to match with existing vehicles
            matched_id = None
            for vid, vinfo in vehicles.items():
                if "last_box" in vinfo and iou(detected_box, vinfo["last_box"]) > IOU_THRESHOLD:
                    matched_id = vid
                    break

            if matched_id is None:
                # New vehicle
                matched_id = next_id
                vehicles[matched_id] = {"first_seen": current_time,
                                        "last_box": detected_box,
                                        "stationary": False}
                next_id += 1
            else:
                # Update existing
                vinfo = vehicles[matched_id]
                if iou(detected_box, vinfo["last_box"]) > IOU_THRESHOLD:
                    # Still in same place
                    elapsed = current_time - vinfo["first_seen"]
                    if elapsed > STATIONARY_THRESHOLD:
                        vinfo["stationary"] = True
                else:
                    # Vehicle moved → reset timer
                    vinfo["first_seen"] = current_time
                    vinfo["stationary"] = False

                vinfo["last_box"] = detected_box

            # Draw box
            color = (0, 0, 255) if vehicles[matched_id]["stationary"] else (0, 255, 0)
            label = f"ID {matched_id}"
            if vehicles[matched_id]["stationary"]:
                label += " 🚨 PARKED"

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Save and display
        out.write(frame)
        cv2.imshow("Illegal Parking Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
