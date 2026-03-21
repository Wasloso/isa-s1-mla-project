# Neural Networks from scratch(?)
### MSc ISA | Machine Learning Algorithms Project

# Overview
TODO

# Dataset: Human Activity Recognition (HAR)
This project uses the [Human Activity Recognition Dataset](https://www.kaggle.com/datasets/arashnic/har-1?) featuring 9,185 subsamples of inertial sensor data.

### Activities
The dataset includes 18 activities:

| Activity | Meaning |
|----------|----------|
| Stand | Standing still |
| Sit | Sitting still |
| Talk-sit | Talking while sitting |
| Talk-stand | Talking while standing or walking |
| Stand-sit | Transition from standing to sitting |
| Lay | Laying still |
| Lay-stand | Transition from laying to standing |
| Pick | Picking up an object |
| Jump | Jumping |
| Push-up | Performing full push-ups |
| Sit-up | Performing sit-ups |
| Walk | Walking |
| Walk-backward | Walking backward |
| Walk-circle | Walking in a circular path |
| Run | Running |
| Stair-up | Ascending stairs |
| Stair-down | Descending stairs |
| Table-tennis | Playing table tennis |



### Data Structure
Each row is a flattened vector of 9,003 columns:
| Range | Sensor / Axis | Description |
|-------|---------------|-------------|
| 0:1500 | Acc (X) | Accelerometer X-axis time-series |
| 1500:3000 | Acc (Y) | Accelerometer Y-axis time-series |
| 3000:4500 | Acc (Z) | Accelerometer Z-axis time-series |
| 4500:6000 | Gyro (X) | Gyroscope X-axis time-series |
| 6000:7500 | Gyro (Y) | Gyroscope Y-axis time-series |
| 7500:9000 | Gyro (Z) | Gyroscope Z-axis time-series |
| 9000 | Label | Class ID (0–17) |
| 9001 | Length | Actual signal duration (before zero-padding) |
| 9002 | Serial | Subsample ID |

# Setup

This project uses [uv](https://astral.sh/uv) as the package manager.

1. **Install uv**:
[Installation instructions](https://docs.astral.sh/uv/getting-started/installation/)

2. **Initialize Environment**:
```bash
git clone <repository-url>
cd <project-directory>
uv sync
```




