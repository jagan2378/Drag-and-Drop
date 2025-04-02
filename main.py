import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Standard laptop width
cap.set(4, 720)   # Standard laptop height

detector = HandDetector(detectionCon=0.8)

# Load images and check if they are loaded correctly - Adjusted positions to be more centered
images = [
    {"img": cv2.imread("D:/Drag_and_Drop/image/car.png", cv2.IMREAD_UNCHANGED), "pos": [200, 100], "placed": False},
    {"img": cv2.imread("D:/Drag_and_Drop/image/cup.png", cv2.IMREAD_UNCHANGED), "pos": [450, 100], "placed": False},
    {"img": cv2.imread("D:/Drag_and_Drop/image/bike.png", cv2.IMREAD_UNCHANGED), "pos": [700, 100], "placed": False},
    {"img": cv2.imread("D:/Drag_and_Drop/image/dog.png", cv2.IMREAD_UNCHANGED), "pos": [950, 100], "placed": False},
]

# Verify images
for item in images:
    if item["img"] is None:
        print("Error loading image:", item)

# Define drop zones with text labels in different order
drop_zones = [
    {"pos": [950, 500], "color": (0, 0, 255), "matched": False, "for_item": 0, "label": "Car"},
    {"pos": [700, 500], "color": (0, 0, 255), "matched": False, "for_item": 1, "label": "Cup"},
    {"pos": [450, 500], "color": (0, 0, 255), "matched": False, "for_item": 2, "label": "Bike"},
    {"pos": [200, 500], "color": (0, 0, 255), "matched": False, "for_item": 3, "label": "Dog"},
]

# Resize images to a good size
for item in images:
    item["img"] = cv2.resize(item["img"], (180, 180))  # Increased size for better visibility

selected_image = None  # Track selected image

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture video. Exiting...")
            break
            
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, flipType=False)

        if hands:
            lmList = hands[0]['lmList']
            cursor_x, cursor_y = lmList[8][0:2]
            
            # Draw cursor point for debugging
            cv2.circle(img, (cursor_x, cursor_y), 5, (255, 0, 0), cv2.FILLED)

            # Check pinch gesture
            length, info, img = detector.findDistance(lmList[8][0:2], lmList[12][0:2], img)
            
            # Debug info - show pinch distance
            cv2.putText(img, f"Pinch: {int(length)}px", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            if length < 50:  # Pinch threshold
                for i, item in enumerate(images):
                    if not item["placed"]:
                        x, y = item["pos"]
                        h, w, _ = item["img"].shape

                        # Hit detection
                        if x < cursor_x < x + w and y < cursor_y < y + h:
                            item["pos"] = [cursor_x - w//2, cursor_y - h//2]
                            selected_image = i

            # Drop logic
            elif selected_image is not None:
                img_x, img_y = images[selected_image]["pos"]
                h, w, _ = images[selected_image]["img"].shape

                for j, zone in enumerate(drop_zones):
                    zx, zy = zone["pos"]
                    # Check if it's the correct zone for this item
                    if (zx - w//2 < img_x < zx + w//2 and 
                        zy - h//2 < img_y < zy + h//2 and 
                        zone["for_item"] == selected_image):
                        images[selected_image]["pos"] = [zx, zy]
                        images[selected_image]["placed"] = True
                        zone["matched"] = True
                        zone["color"] = (0, 255, 0)  # Change to green only if correct match
                selected_image = None

        # Draw Drop Zones - Using text labels instead of rectangles
        for zone in drop_zones:
            zx, zy = zone["pos"]
            # Draw text label
            text_size = cv2.getTextSize(zone["label"], cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
            text_x = zx - text_size[0]//2  # Center text horizontally
            cv2.putText(img, zone["label"], (text_x, zy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, zone["color"], 3)

        # Overlay draggable images
        for item in images:
            img = cvzone.overlayPNG(img, item["img"], item["pos"])

        cv2.imshow("Drag and Drop Game", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Program stopped by user")
            break

except KeyboardInterrupt:
    print("\nProgram stopped by user")
finally:
    cap.release()
    cv2.destroyAllWindows()