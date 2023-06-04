import pandas as pd
import cv2
import glob
import numpy as np

# Load the ground truth dataframe from a CSV file
gt_df = pd.read_csv(r"C:\INTERNSHIP\CSV Annotations\CATDOG\gt\gt.txt", header=None)
gt_df.columns = ['A', 'B', 'x', 'y', 'w', 'h', 'C', 'D', 'E']
gt_df.drop(columns=['A', 'B', 'C', 'D', 'E'], inplace=True)
gt_df['image_id'] = ['0', '1', '2', '2', '2', '2', '3', '3', '3', '3', '3', '3', '4', '5', '5', '6', '7', '8', '9']

# Load the predicted dataframe from a CSV file
pred_df = pd.read_csv(r"C:\INTERNSHIP\CSV Annotations\CATDOG\pred.txt", header=None)
pred_df.columns = ['A', 'B', 'x', 'y', 'w', 'h', 'C', 'D', 'E']
pred_df.drop(columns=['A', 'B', 'C', 'D', 'E'], inplace=True)
pred_df['image_id'] = ['0', '1', '2', '2', '2', '2', '3', '3', '3', '3', '3', '3', '4', '5', '5', '6', '7', '8', '9']


# Function to calculate IoU and overlap
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    gt_area = box1[2] * box1[3]
    pred_area = box2[2] * box2[3]
    union_area = gt_area + pred_area - inter_area
    iou = inter_area / union_area
    overlap = inter_area / max(gt_area, pred_area)
    return iou, overlap


# Path to the directory containing the images
image_directory = "C:/INTERNSHIP/images/*.jpeg"

# Get the list of image file paths
image_files = glob.glob(image_directory)

# Check if there are images in the directory
if len(image_files) == 0:
    print("No image files found in the directory.")
    exit()

# Initialize the confusion matrix
confusion_matrix = np.zeros((2, 2))  # 2x2 matrix for binary classification

# Iterate over each image
for index, image_path in enumerate(image_files):
    # Load the current image
    image = cv2.imread(image_path)

    # Draw bounding boxes for ground truth annotations
    for _, row in gt_df[gt_df['image_id'] == str(index)].iterrows():
        x = row['x']
        y = row['y']
        w = row['w']
        h = row['h']
        p1 = (int(x), int(y))
        p2 = (int(x + w), int(y + h))
        cv2.rectangle(image, p1, p2, (0, 0, 255), thickness=2)

    # Draw bounding boxes for predicted annotations
    for _, row in pred_df[pred_df['image_id'] == str(index)].iterrows():
        x = row['x']
        y = row['y']
        w = row['w']
        h = row['h']
        p1 = (int(x), int(y))
        p2 = (int(x + w), int(y + h))
        cv2.rectangle(image, p1, p2, (0, 255, 0), thickness=2)

    # Calculate precision and recall for each predicted bounding box
    pred_boxes = pred_df[pred_df['image_id'] == str(index)][['x', 'y', 'w', 'h']].values
    gt_boxes = gt_df[gt_df['image_id'] == str(index)][['x', 'y', 'w', 'h']].values

    for i, pred_box in enumerate(pred_boxes):
        overlaps = []
        for gt_box in gt_boxes:
            _, overlap = calculate_iou(gt_box, pred_box)
            overlaps.append(overlap)

        # Calculate IoU for each ground truth box and the predicted box
        iou_list = [calculate_iou(gt_box, pred_box)[0] for gt_box in gt_boxes]

        # Find the maximum IoU and its corresponding index
        max_iou = max(iou_list)
        max_iou_index = iou_list.index(max_iou)

        # Determine if the prediction is a true positive or false positive
        if max_iou >= 0.5:
            confusion_matrix[1, 1] += 1  # True positive
        else:
            confusion_matrix[0, 1] += 1  # False positive

        # Remove the matched ground truth box from the list
        gt_boxes = np.delete(gt_boxes, max_iou_index, axis=0)

        # Add the remaining unmatched ground truth boxes as false negatives
        confusion_matrix[0, 0] += gt_boxes.shape[0]  # False negative

        print(f"Image {index + 1}, Bounding Box {i + 1}:")
        print("Overlap:", max(overlaps))
        print("Overlap Percentage:", round(max(overlaps) * 100, 2), "%")
        print("TP:", confusion_matrix[1, 1])
        print("FP:", confusion_matrix[0, 1])
        print("TN:", confusion_matrix[1, 0])
        print("FN:", confusion_matrix[0, 0], "\n")

    # Display the image
    cv2.imshow("Image", image)

    # Check if the user pressed the 'q' key
    if cv2.waitKey(2000) & 0xFF == ord('q'):
        break

    # Close the image window
    cv2.destroyAllWindows()

# Calculate precision and recall from the confusion matrix
true_positives = confusion_matrix[1, 1]
false_positives = confusion_matrix[0, 1]
true_negatives = confusion_matrix[1, 0]
false_negatives = confusion_matrix[0, 0]

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0

print("Overall Precision:", precision)
print("Overall Recall:", recall)
