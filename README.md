# DTAP: Dynamic Task Adjustment and Planning

DTAP is a **memory-augmented robot navigation framework** that enables continuous execution of natural language instructions. It builds a value‑filtered memory from RGB‑D streams and supports asynchronous querying for dynamic task adjustment.

---

## 📖 Overview

DTAP addresses the limitation of traditional blocking execution in robotic systems by introducing a non‑blocking memory and planning architecture. The robot continuously constructs a long‑term spatial memory from its RGB‑D camera stream, storing only segments that exceed a learned **value score** (based on saliency, novelty, and eventfulness). While the robot executes its current motion policy, memory building runs in the background. When a new instruction arrives, DTAP queries the memory asynchronously, retrieves relevant experiences, and adjusts the task without interrupting the ongoing control.

This repository contains the complete implementation, including:

- Value-based memory construction from RGB-D sequences
- Retrieval-augmented question answering using LLMs (via Ollama)
- Integration with Milvus vector database
- ROS 2 compatible nodes for simulation and real robot deployment

---

## ✨ Key Features

- **Value‑based memory filtering** – stores only informative video clips based on saliency, novelty, and events.
- **RGB‑D integration** – depth improves saliency estimation and filters distant objects.
- **Retrieval‑augmented QA** – query memory by text, position, or time; LLM iterates until answer.
- **Non‑blocking control** – memory construction runs in background; queries do not interrupt robot motion.
- **SLAM‑ready** – works with real‑time pose estimates (ROS 2 odometry, ORB‑SLAM).
- **Simulation support** – fully compatible with Gazebo and NVIDIA Isaac Sim.

---

## 🛠 System Architecture

The system consists of two main modules:

### 1. Memory Builder

- **Input:** RGB-D image stream + robot poses (timestamped)
- **Process:** Splits stream into temporal clips, generates captions via VLM, computes value score, and stores high-value clips in Milvus.
- **Value Score Components:**
  - *Saliency*: RGB + depth differences between first and last frame.
  - *Novelty*: New objects detected (with depth-based distance filtering).
  - *Event*: Keyword or LLM-based detection of important events.

### 2. Memory Query

- **Input:** Natural language question
- **Process:** LLM can call retrieval functions (text, position, time), retrieves candidates from Milvus, re-ranks using similarity + value, and generates final answer.
- **Output:** JSON with answer, reasoning, and used memory IDs.

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**

- **Milvus** (standalone or Docker): [Installation guide](https://milvus.io/docs/install_standalone-docker.md)

- **Ollama** with models:

  ```bash
  ollama pull llava:7b          # VLM for captioning
  ollama pull nomic-embed-text  # Embedding model
  ollama pull llama3.1:8b       # LLM for query reasoning
  ```


- **System dependencies** (for Ubuntu):

  bash

```

  sudo apt update
  sudo apt install python3-opencv

```


### Installation

Clone the repository and install Python packages:

bash

  ```

git clone https://github.com/yourusername/DTAP.git
cd DTAP
pip install -r requirements.txt

  ```


### Dataset Preparation (TUM RGB-D Format)

Organize your data as follows:

text

```

data/
├── rgb.txt                 # timestamp filename
├── depth.txt               # timestamp filename
├── groundtruth.txt         # timestamp tx ty tz qx qy qz qw (optional)
├── rgb/                    # RGB images (PNG)
└── depth/                  # Depth images (PNG, 16-bit mm)

```


Example `rgb.txt`:

text

```

# timestamp filename

1311876799.7716 rgb/1311876799.7716.png
...

```


### Build Memory from Dataset

bash

```

python main.py --mode build \
    --rgb_txt data/rgb.txt \
    --depth_txt data/depth.txt \
    --groundtruth data/groundtruth.txt

```


### Query the Memory

bash

```

python main.py --mode query --question "Where did I see a red chair?"

```


Output example:

json

```

{
  "answer": "You saw a red chair near the entrance at timestamp 1311876805.2.",
  "reasoning": "Retrieved memory ID 42 has caption 'red chair near entrance' and matches query.",
  "used_memories": [42]
}

```


### Run Demo Script

bash

```

python demo.py

```


------

## ⚙️ Configuration

All parameters are in `config.py`. Important ones:

| Parameter             | Description                                         | Default |
| :-------------------- | :-------------------------------------------------- | :------ |
| `CLIP_DURATION`       | Duration of each memory clip (seconds)              | 3.0     |
| `WEIGHT_SALIENCY`     | Weight for saliency in value score                  | 0.3     |
| `WEIGHT_NEW_OBJECTS`  | Weight for novelty                                  | 0.4     |
| `WEIGHT_EVENT`        | Weight for eventfulness                             | 0.3     |
| `VALUE_THRESHOLD`     | Minimum value to store a clip                       | 0.5     |
| `OBJECT_MAX_DISTANCE` | Only consider objects within this distance (meters) | 5.0     |
| `RETRIEVAL_K`         | Number of candidates retrieved initially            | 20      |
| `FINAL_K`             | Number of memories returned after re-ranking        | 5       |
| `MAX_ITERATIONS`      | Max LLM retrieval steps per query                   | 3       |

------

## 🤖 ROS 2 Simulation

DTAP can run in a simulated environment using ROS 2 and Gazebo. A dedicated ROS node subscribes to RGB, depth, and odometry topics, and builds memory online.

### Launch Simulation (TurtleBot3 example)

bash

```

# Terminal 1: Start Gazebo world

ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Terminal 2: Start memory builder node

ros2 run dtap memory_builder_node --ros-args -p clip_duration:=3.0

# Terminal 3: Start query server

python scripts/query_server.py

```


Send queries via HTTP:

bash

```

curl -X POST http://localhost:5000/query -H "Content-Type: application/json" -d '{"question": "Where is the table?"}'

```


------

## 📁 Project Structure

text

```

.
├── config.py               # Configuration parameters
├── memory_builder.py       # Memory construction pipeline
├── memory_query.py         # Retrieval-augmented QA
├── value_scorer.py         # Value computation (saliency, novelty, event)
├── utils.py                # Helper functions (TUM loading, VLM calls, embeddings, Milvus init)
├── main.py                 # CLI entry point
├── demo.py                 # Demonstration script
├── requirements.txt        # Python dependencies
├── scripts/
│   └── query_server.py     # Flask server for HTTP queries
├── docs/
│   └── simulation.md       # Detailed simulation guide
└── README.md               # This file

```


------

## 📄 Citation

If you use this code in your research, please cite our paper:

bibtex

```

@article{dtap2025,
  title={Dynamic Task Adjustment and Planning (DTAP): A Memory-Augmented Navigation Framework},
  author={Your Name and Colleagues},
  journal={arXiv preprint},
  year={2025}
}

```


------

## 🙏 Acknowledgments

- [Milvus](https://milvus.io/) – vector database
- [Ollama](https://ollama.com/) – local LLM inference
- [TUM RGB-D dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) – data format inspiration
- NVIDIA Isaac ROS and Jetson community
```
