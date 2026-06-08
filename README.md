YOLO-based Visual Concept Retrieval
This project implements an image retrieval system that searches images using visual concepts detected by YOLO.  
The system supports text queries such as `person horse`, `car bus`, or Vietnamese queries such as `người và con ngựa`.
The main idea is:
```text
Image subset
    ↓
YOLO object detection
    ↓
YOLO concept index
    ↓
Text query → query concepts
    ↓
Weighted concept matching
    ↓
Top-K retrieved images
    ↓
Evaluation with Visual Genome ground truth
```
Project Overview
This repository contains a YOLO-based image retrieval pipeline. Instead of using image captions or manual annotations for retrieval, the system uses YOLO to automatically detect objects in each image and converts them into searchable visual concepts.
Visual Genome annotations are only used as ground truth for evaluation. They are not used to build the retrieval index.
Repository Files
```text
.
├── YOLO (1).ipynb        # Build YOLO concept index, search, and evaluation
├── Demo_IR.ipynb         # Run the Flask demo application on Google Colab
└── README.md
```
Recommended rename for cleaner GitHub structure:
```text
YOLO (1).ipynb  →  YOLO_Visual_Concept_Retrieval.ipynb
Demo_IR.ipynb   →  Run_Demo_App.ipynb
```
Features
Detect objects in images using YOLOv8.
Build a YOLO concept index for image retrieval.
Support English and Vietnamese text queries.
Translate Vietnamese queries to English.
Normalize query concepts using rules, synonyms, plural handling, and semantic embedding fallback.
Search images using weighted concept matching.
Evaluate Top-K retrieval performance using Visual Genome object annotations.
Run a Flask-based demo app from Google Colab.
Tech Stack
Python
Google Colab
YOLOv8 / Ultralytics
Visual Genome dataset
NumPy
Pandas
Matplotlib
Pillow
tqdm
deep-translator
SentenceTransformers
Flask
Dataset
The project uses a Visual Genome image subset.
Expected Google Drive structure:
```text
IR_Photo_Gallery_Project/
├── app/
│   └── app4.py
└── visual_genome/
    ├── annotations/
    │   └── objects.json
    ├── common_images/
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   └── ...
    └── indexes/
        ├── index_mapping_common.csv
        ├── yolo_concept_index_30k.json
        ├── yolo_concept_idf_30k.json
        └── vg_ground_truth_index_30k.json
```
The mapping file should contain the following columns:
```text
global_id, standard_filename, image_id, original_filename,
original_folder, original_relative_path, common_relative_path
```
Methodology
1. YOLO Concept Extraction
Each image is processed by YOLOv8. Detected objects are converted into normalized visual concepts.
Example output:
```text
Detected concepts: ['bench', 'car', 'person', 'train']
Concept counts: {'train': 1, 'person': 4, 'car': 3, 'bench': 1}
```
2. YOLO Concept Index
The detected concepts are saved into a JSON index:
```text
indexes/yolo_concept_index_30k.json
```
Each indexed image contains:
image ID
file path information
detected concepts
concept counts
number of objects
bounding box detections
The notebook supports resume mode, so already indexed images are skipped when the notebook is rerun.
3. Concept IDF Weighting
The system calculates IDF scores for YOLO concepts:
```text
idf = log((N + 1) / (df + 1)) + 1
```
This gives higher weight to rarer and more informative concepts.
Example frequent concepts in the indexed subset:
```text
person, car, chair, table, cup
```
4. Query Processing
The system supports both English and Vietnamese queries.
Example:
```text
Input query: người và con ngựa
Translated query: people and horses
Mapped concepts: ['person', 'horse']
```
Query processing includes:
Vietnamese-to-English translation.
Lowercasing and text normalization.
Stopword removal.
Rule-based matching.
Synonym and plural normalization.
SentenceTransformer fallback for semantic concept mapping.
The embedding model is only used to map query phrases to YOLO classes. It is not used as a CLIP-style image-text retrieval model.
5. Image Retrieval
Images are ranked using weighted concept matching.
The score considers:
matched query concepts
concept frequency in each image
IDF weight of each concept
coverage of query concepts
Images matching more query concepts receive higher scores.
6. Evaluation
Visual Genome `objects.json` is used as ground truth.
An image is considered relevant if its ground truth concepts contain all query concepts.
Evaluation metrics:
Precision@K
Recall@K
Average Precision@K
Current Experiment Result
The current notebook run indexed 3,000 images and evaluated retrieval on multiple queries.
Summary result:
```text
P@1   = 0.5333
P@5   = 0.5200
P@10  = 0.5133
R@1   = 0.0137
R@5   = 0.1079
R@10  = 0.1728
AP@10 = 0.4132
```
The notebook is configured with:
```python
MAX_IMAGES_TO_INDEX = 3000
CONF_THRESHOLD = 0.25
SAVE_EVERY = 200
```
To run on the full 30K subset, set:
```python
MAX_IMAGES_TO_INDEX = None
```
How to Run
1. Open the notebook in Google Colab
Open:
```text
YOLO (1).ipynb
```
2. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
3. Install dependencies
```python
!pip install -q ultralytics deep-translator sentence-transformers
```
4. Check dataset paths
The notebook expects:
```python
VG_DIR = "/content/drive/MyDrive/IR_Photo_Gallery_Project/visual_genome"
```
Update this path if your dataset is stored elsewhere.
5. Build YOLO index
Run the notebook cells from top to bottom to:
load image mapping,
verify image paths,
load YOLO,
detect objects,
build the YOLO concept index,
build IDF scores,
build Visual Genome ground truth,
run search,
evaluate Top-K results.
6. Test a query
Example:
```python
output = search_yolo_visual_concept(
    query="người và con ngựa",
    yolo_index=yolo_index,
    concept_idf=yolo_idf,
    top_k=10,
    language="auto",
    use_embedding=True
)

show_search_results(output)
```
Running the Demo App
The `Demo_IR.ipynb` notebook is used to run the Flask demo from Google Colab.
1. Install dependencies
```python
!pip install -q flask deep-translator sentence-transformers transformers accelerate pillow tqdm pandas matplotlib faiss-cpu
```
2. Serve the Colab port
```python
from google.colab import output
output.serve_kernel_port_as_window(5000)
```
3. Start Flask app
```python
APP_DIR = "/content/drive/MyDrive/IR_Photo_Gallery_Project/app"

!cd "$APP_DIR" && nohup python app4.py > flask.log 2>&1 &
```
4. Stop Flask app
```python
!pkill -f app4.py
```
Output Files
The main output files are:
```text
indexes/
├── index_mapping_common.csv
├── yolo_concept_index_30k.json
├── yolo_concept_idf_30k.json
└── vg_ground_truth_index_30k.json
```
Important Notes
YOLO is used to build the actual retrieval index.
Visual Genome annotations are used only for evaluation.
Vietnamese queries are translated and normalized before retrieval.
The system retrieves images based on object-level visual concepts, not full natural-language scene understanding.
Current search quality depends on YOLO detection quality and the limited COCO object vocabulary.
Limitations
YOLOv8 detects only predefined COCO object classes.
The system may miss abstract concepts, actions, attributes, or relationships between objects.
Query understanding is limited to object-level concepts.
Recall can be low when the query contains rare objects or concepts not covered by YOLO.
Visual Genome has a much broader vocabulary than YOLO, so evaluation requires label normalization.
Future Improvements
Use CLIP or another vision-language model for more flexible text-image retrieval.
Combine YOLO concept matching with image embeddings.
Add support for attributes such as color, size, and action.
Improve Vietnamese query processing.
Build a larger index on the full 30K image subset.
Add a web interface for uploading new images and searching interactively.
Project Summary
This project demonstrates an image retrieval pipeline based on automatic object detection. YOLO is used to generate searchable visual concepts from images, while Visual Genome annotations are used only to evaluate retrieval quality. The system supports Vietnamese and English queries and ranks results using weighted concept matching.
