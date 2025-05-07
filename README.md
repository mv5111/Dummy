import os
import cv2
import faiss
import yaml
import json
import base64
import numpy as np
from PIL import Image
from datetime import datetime
from typing import List, Dict, Tuple
from huggingface_hub import InferenceClient
from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq

# ====================
# Configuration Loader
# ====================

class WorkflowCreatorConfig:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.video_path = os.path.join(os.path.dirname(config_path), config["video_source"])
        self.output_path = os.path.join(os.path.dirname(config_path), config["output_json"])
        self.frame_interval = config.get("frame_interval", 30)
        self.min_confidence = config.get("min_confidence", 0.8)
        self.models = config["models"]
        self.faiss_config = config.get("faiss", {})
        self.screenshot_dir = os.path.join(os.path.dirname(config_path), config.get("screenshot_dir", "screenshots"))

        os.makedirs(self.screenshot_dir, exist_ok=True)

# ====================
# Image Embedding & FAISS
# ====================

class ImageVectorStore:
    def __init__(self, config: WorkflowCreatorConfig):
        self.index = None
        self.embeddings = {}
        self.model = InferenceClient(config.models["embedding"])
        self.dimension = 384  # Dimension for 'all-MiniLM-L6-v2'
        self.similarity_threshold = config.faiss_config.get("similarity_threshold", 0.85)

    def initialize_index(self):
        self.index = faiss.IndexFlatL2(self.dimension)

    def add_embedding(self, image: Image.Image, scene_id: str):
        embedding = self._get_embedding(image)
        if self.index is None:
            self.initialize_index()

        self.embeddings[scene_id] = embedding
        self.index.add(np.array([embedding]).astype('float32'))

    def find_similar(self, image: Image.Image) -> Tuple[str, float]:
        if self.index is None or len(self.embeddings) == 0:
            return None, 0.0

        query_embed = self._get_embedding(image)
        distances, indices = self.index.search(np.array([query_embed]).astype('float32'), 1)

        if indices[0][0] == -1:
            return None, 0.0

        scene_id = list(self.embeddings.keys())[indices[0][0]]
        similarity = 1 - (distances[0][0] / self.dimension)
        return (scene_id, similarity) if similarity > self.similarity_threshold else (None, 0.0)

    def _get_embedding(self, image: Image.Image) -> List[float]:
        image = image.convert("RGB")
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        response = self.model.feature_extraction(images=[img_str])
        return response[0]

# ====================
# Frame Analysis
# ====================

class FrameAnalyzer:
    def __init__(self, config: WorkflowCreatorConfig):
        self.model_name = config.models["screen_analysis"]
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_name)

    def analyze_frame(self, frame: Image.Image) -> Dict:
        try:
            inputs = self.processor(images=frame, return_tensors="pt")
            outputs = self.model.generate(**inputs)
            description = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            analysis = json.loads(description)
            return self._validate_analysis(analysis)
        except Exception as e:
            print(f"Analysis error: {e}")
            return self._create_empty_analysis()

    def _validate_analysis(self, analysis: Dict) -> Dict:
        required_sections = ["metadata", "structure", "components"]
        return analysis if all(section in analysis for section in required_sections) else self._create_empty_analysis()

    def _create_empty_analysis(self) -> Dict:
        return {
            "metadata": {"screen_type": "unknown"},
            "structure": [],
            "components": [],
            "text_elements": [],
            "visual_hierarchy": []
        }

# ====================
# Scene Management
# ====================

class SceneManager:
    def __init__(self, vector_store: ImageVectorStore, config: WorkflowCreatorConfig):
        self.vector_store = vector_store
        self.config = config
        self.scenes = []
        self.current_scene = None
        self.initial_scene_id = None

    def detect_scene_change(self, frame: Image.Image) -> bool:
        if not self.current_scene:
            return True

        scene_id, similarity = self.vector_store.find_similar(frame)
        return scene_id != self.current_scene['scene_id']

    def create_new_scene(self, frame: Image.Image, analysis: Dict):
        scene_id = f"scene_{len(self.scenes)+1}"
        self.vector_store.add_embedding(frame, scene_id)

        self.current_scene = {
            "scene_id": scene_id,
            "screenshot": self._store_screenshot(frame, scene_id),
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "analysis": analysis,
            "actions": []
        }

        if not self.initial_scene_id:
            self.initial_scene_id = scene_id

        self.scenes.append(self.current_scene)

    def finalize_scene(self):
        if self.current_scene:
            self.current_scene['end_time'] = datetime.now().isoformat()

    def check_completion(self, frame: Image.Image) -> bool:
        if not self.initial_scene_id:
            return False

        scene_id, similarity = self.vector_store.find_similar(frame)
        return scene_id == self.initial_scene_id and len(self.scenes) > 1

    def _store_screenshot(self, frame: Image.Image, scene_id: str) -> str:
        path = os.path.join(self.config.screenshot_dir, f"{scene_id}.png")
        frame.save(path)
        return path

# ====================
# Video Processing
# ====================

class VideoProcessor:
    def __init__(self, config: WorkflowCreatorConfig):
        self.config = config
        self.cap = cv2.VideoCapture(config.video_path)
        self.vector_store = ImageVectorStore(config)
        self.frame_analyzer = FrameAnalyzer(config)
        self.scene_manager = SceneManager(self.vector_store, config)
        self.frame_count = 0

    def process_video(self) -> Dict:
        workflow = {
            "metadata": self._create_metadata(),
            "scenes": [],
            "embeddings_index": "faiss_index.bin",
            "completed": False
        }

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            if self.frame_count % self.config.frame_interval == 0:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                analysis = self.frame_analyzer.analyze_frame(pil_image)

                if self.scene_manager.detect_scene_change(pil_image):
                    self.scene_manager.finalize_scene()
                    self.scene_manager.create_new_scene(pil_image, analysis)
                else:
                    self._update_current_scene()

                if self.scene_manager.check_completion(pil_image):
                    workflow['completed'] = True
                    break

            self.frame_count += 1

        self._finalize_processing(workflow)
        return workflow

    def _update_current_scene(self):
        # Placeholder for action tracking logic
        pass

    def _finalize_processing(self, workflow: Dict):
        self.cap.release()
        faiss.write_index(self.vector_store.index, "faiss_index.bin")
        workflow['scenes'] = [s.copy() for s in self.scene_manager.scenes]

    def _create_metadata(self) -> Dict:
        return {
            "created_at": datetime.now().isoformat(),
            "video_source": self.config.video_path,
            "config": vars(self.config)
        }

# ====================
# Workflow Generation
# ====================

class WorkflowGenerator:
    def generate(self, processed_data: Dict) -> Dict:
        return {
            "workflow": {
                "metadata": processed_data["metadata"],
                "scenes": self._process_scenes(processed_data["scenes"]),
                "embeddings": processed_data["embeddings_index"],
                "completion": processed_data["completed"]
            },
            "statistics": {
                "total_scenes": len(processed_data["scenes"]),
                "unique_screens": len({s['scene_id'] for s in processed_data["scenes"]})
            }
        }

    def _process_scenes(self, scenes: List) -> List:
        return [{
            "id": s["scene_id"],
            "screen_type": s["analysis"]["metadata"]["screen_type"],
            "duration": self._calculate_duration(s["start_time"], s["end_time"]),
            "screenshot": s["screenshot"],
            "structure": s["analysis"]["structure"],
            "components": self._format_components(s["analysis"]["components"]),
            "text_elements": s["analysis"]["text_elements"],
            "actions": s["actions"]
        } for s in scenes
::contentReference[oaicite:0]{index=0}
 
import sys
from io import BytesIO

def main():
    config_path = os.path.join("workflow creation", "config.yaml")
    config = WorkflowCreatorConfig(config_path)

    processor = VideoProcessor(config)
    processed_data = processor.process_video()

    generator = WorkflowGenerator()
    workflow_output = generator.generate(processed_data)

    with open(config.output_path, 'w') as f:
        json.dump(workflow_output, f, indent=4)

    print(f"Workflow JSON saved to {config.output_path}")

if __name__ == "__main__":
    main()
