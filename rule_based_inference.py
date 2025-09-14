from shapely.geometry import Polygon, Point

# Define no-parking zones as polygons (example coordinates)
no_parking_zones = [
    Polygon([(100,100), (300,100), (300,300), (100,300)])  # rectangular zone
]

def is_in_no_parking_zone(x1, y1, x2, y2):
    # Take vehicle center point
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    point = Point(cx, cy)

    for zone in no_parking_zones:
        if zone.contains(point):
            return True
    return False

# Example detections list for testing
detections = [
    {"x1": 120, "y1": 120, "x2": 180, "y2": 180, "class": "car"},
    {"x1": 400, "y1": 400, "x2": 450, "y2": 450, "class": "bus"},
    {"x1": 150, "y1": 150, "x2": 200, "y2": 200, "class": "truck"},
]

# During inference loop
for det in detections:  # x1,y1,x2,y2,class
    if det["class"] in ["car", "bus", "truck"]:
        if is_in_no_parking_zone(det["x1"], det["y1"], det["x2"], det["y2"]):
            print("🚨 Illegal Parking Detected!")
