%pip install opencv-python ultralytics
import os
import yaml
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import pytesseract
import xml.etree.ElementTree as ET
import shutil

class ChequeProcessor:
    def __init__(self, xml_path, images_dir, output_dir):
        self.xml_path = xml_path
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.main_classes = ['dollar_amount_section', 'legal_amount_section', 'date_section']
        os.makedirs(os.path.expanduser(output_dir), exist_ok=True)

    def xml_to_yolo_stage1(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        main_class_map = {cls: idx for idx, cls in enumerate(self.main_classes)}
        
        # Output folders
        train_img_dir = os.path.join(self.output_dir, 'stage1', 'images', 'train')
        val_img_dir = os.path.join(self.output_dir, 'stage1', 'images', 'val')
        train_label_dir = os.path.join(self.output_dir, 'stage1', 'labels', 'train')
        val_label_dir = os.path.join(self.output_dir, 'stage1', 'labels', 'val')
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(val_label_dir, exist_ok=True)

        images_processed = 0
        for image in root.findall('image'):
            img_file = image.get('file')
            if img_file is None:
                continue
            img_path = os.path.join(self.images_dir, img_file)
            if not os.path.exists(img_path):
                continue

            img_width = int(image.get('width'))
            img_height = int(image.get('height'))

            if np.random.rand() < 0.8:
                img_dest = os.path.join(train_img_dir, img_file)
                label_dest = os.path.join(train_label_dir, f"{os.path.splitext(img_file)[0]}.txt")
            else:
                img_dest = os.path.join(val_img_dir, img_file)
                label_dest = os.path.join(val_label_dir, f"{os.path.splitext(img_file)[0]}.txt")

            shutil.copy(img_path, img_dest)

            with open(label_dest, 'w') as lf:
                for box in image.findall('box'):
                    label = box.get('label')
                    if label in self.main_classes:
                        x_min = float(box.get('xtl'))
                        y_min = float(box.get('ytl'))
                        x_max = float(box.get('xbr'))
                        y_max = float(box.get('ybr'))
                        x_center = (x_min + x_max) / 2 / img_width
                        y_center = (y_min + y_max) / 2 / img_height
                        width = (x_max - x_min) / img_width
                        height = (y_max - y_min) / img_height
                        class_id = main_class_map[label]
                        lf.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
            images_processed += 1
        
        print(f"Total images processed: {images_processed}")

    def create_yolo_configs(self):
        stage1_config = {
            'path': os.path.join(self.output_dir, 'stage1'),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.main_classes),
            'names': self.main_classes
        }
        with open(os.path.join(self.output_dir, 'stage1_config.yaml'), 'w') as f:
            yaml.dump(stage1_config, f)

class ChequeTrainingPipeline:
    def __init__(self, processor):
        self.processor = processor
        self.stage1_model = None

    def train_stage1(self, epochs=50, batch_size=8):
        config_path = os.path.join(self.processor.output_dir, 'stage1_config.yaml')
        model = YOLO('yolov8s.pt')
        results = model.train(
            data=config_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            device='cpu'  # Use 'cuda' if GPU available
        )
        self.stage1_model = model
        return results

class ChequeInspector:
    def __init__(self, stage1_model_path):
        self.stage1_model = YOLO(stage1_model_path)

    def predict_and_visualize(self, image_path, output_path='output_with_boxes.jpg'):
        img = Image.open(image_path).convert("RGB")
        result = self.stage1_model(image_path)[0]

        draw = ImageDraw.Draw(img)
        for box in result.boxes:
            cls_id = int(box.cls)
            bbox = box.xyxy[0].tolist()
            class_name = self.stage1_model.names[cls_id]
            conf = float(box.conf)
            draw.rectangle(bbox, outline="red", width=2)
            draw.text((bbox[0], bbox[1] - 10), f"{class_name} ({conf:.2f})", fill="yellow")
        
        img.save(output_path)
        print(f"Saved visualization to {output_path}")

# Usage Example
if __name__ == "__main__":
    # Paths
    images_dir = "/Workspace/Users/mrinalini.vettri@fisglobal.com/yolo_check_training/checks"
    annotations_path = "annotations.xml"
    output_dir = "checks"

    # Step 1: Prepare data
    processor = ChequeProcessor(annotations_path, images_dir, output_dir)
    processor.xml_to_yolo_stage1()
    processor.create_yolo_configs()

    # Step 2: Train stage 1
    pipeline = ChequeTrainingPipeline(processor)
    stage1_results = pipeline.train_stage1(epochs=50)

    # Step 3: Visualize prediction
    inspector = ChequeInspector(stage1_model_path='checks/stage1/weights/best.pt')
    test_image = os.path.join(images_dir, '3307435490.tif')  # Replace with any test image
    inspector.predict_and_visualize(test_image, output_path='visualized_prediction.jpg')
