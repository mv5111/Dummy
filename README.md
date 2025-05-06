import os
import cv2
import json
import yaml
import datetime
from PIL import Image
from io import BytesIO
from dataclasses import dataclass
from typing import List, Dict
from huggingface_hub import InferenceClient


@dataclass
class Scene:
    id: str
    screen_type: str
    screenshot: str
    duration: float
    section: str
    structure: List[Dict]
    embeddings: str
    text_elements: List[str]
    actions: List[str]


class VideoProcessor:
    def __init__(self, video_path, scene_interval=5):
        self.video_path = video_path
        self.scene_interval = scene_interval  # seconds

    def extract_scenes(self, output_folder):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        os.makedirs(output_folder, exist_ok=True)

        scenes = []
        for t in range(0, int(duration), self.scene_interval):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                break
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            filename = os.path.join(output_folder, f"scene_{len(scenes)+1}.png")
            image.save(filename)
            scenes.append({
                "id": f"scene_{len(scenes)+1}",
                "image_path": filename,
                "duration": self.scene_interval
            })

        cap.release()
        return scenes


class FrameAnalyzer:
    def __init__(self, model_name="Qwen/Qwen-VL"):
        self.client = InferenceClient(model=model_name)

    def analyze_frame(self, frame: Image.Image):
        buffer = BytesIO()
        frame.save(buffer, format="PNG")
        buffer.seek(0)

        prompt = "Describe the scene in detail and list all the user actions in the UI."
        response = self.client.image_to_text(image=buffer, prompt=prompt)
        return response


class ImageVectorStore:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = InferenceClient(model=model_name)

    def _get_embedding(self, image: Image.Image):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        response = self.model.feature_extraction(buffer)
        return response


def generate_json(video_path, scene_data, output_path, config):
    output = {
        "workflow": {
            "metadata": {
                "created_at": datetime.datetime.now().isoformat(),
                "video_source": os.path.basename(video_path),
                "config": config
            },
            "scenes": [],
            "statistics": {
                "total_scenes": len(scene_data),
                "unique_screens": len(scene_data)
            },
            "completion": True
        }
    }

    for scene in scene_data:
        output["workflow"]["scenes"].append({
            "id": scene.id,
            "screen_type": scene.screen_type,
            "screenshot": scene.screenshot,
            "duration": scene.duration,
            "section": scene.section,
            "structure": scene.structure,
            "embeddings": scene.embeddings,
            "text_elements": scene.text_elements,
            "actions": scene.actions
        })

    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    video_path = config["video_path"]
    scene_output_dir = config["scene_output_dir"]
    output_json_path = config["output_json_path"]

    video_processor = VideoProcessor(video_path)
    frame_analyzer = FrameAnalyzer()
    vector_store = ImageVectorStore()

    scene_infos = video_processor.extract_scenes(scene_output_dir)
    scene_objects = []

    for info in scene_infos:
        image = Image.open(info["image_path"])
        description = frame_analyzer.analyze_frame(image)
        embedding = vector_store._get_embedding(image)

        # Dummy structure based on description
        structure = [{
            "position": [100, 200, 150, 225],
            "components": [{
                "type": "button",
                "position": [100, 200, 150, 225],
                "text": "Save",
                "properties": {
                    "color": "#4287f5",
                    "font": "Arial 12pt"
                },
                "actions": ["click"]
            }]
        }]

        scene_objects.append(Scene(
            id=info["id"],
            screen_type="dashboard",
            screenshot=info["image_path"],
            duration=info["duration"],
            section="main_menu",
            structure=structure,
            embeddings="faiss_index.bin",
            text_elements=[description],
            actions=["click"]
        ))

    generate_json(video_path, scene_objects, output_json_path, config)


if __name__ == "__main__":
    config_path = "/Workspace/Users/mrinalini.vettri@fisglobal.com/video_analysis/workflow creation/config.yaml"
    main(config_path)
