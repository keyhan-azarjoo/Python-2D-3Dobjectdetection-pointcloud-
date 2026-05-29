# 2D & 3D Object Detection on Point-Cloud / LiDAR Data

A deep-learning pipeline for 2D and 3D object detection, semantic segmentation, and digital-twin generation from LiDAR point-cloud data. (MSc dissertation research.)

## What it does
- 3D point-cloud segmentation and classification with **PointNet**
- 2D detection with **CNN** and **YOLO**-family models
- LiDAR spatial processing → semantic segmentation → **digital-twin** generation

## Contents
- `Project Code and Data/` — implementation and datasets
- `Keyhan Azarjoo Project Report.pdf` — the full written report

## How it works
Raw LiDAR point clouds are processed and segmented (PointNet operates directly on unordered 3D points), objects are detected in both 2D and 3D, and the results are assembled into accurate 3D digital twins of the captured assets.

## Role within my work (MyOTGO)
3D perception and spatial mapping are directly relevant to MyOTGO's **autonomous-drone subsystem** — environment understanding, obstacle / asset detection and mapping. This MSc research (Distinction; with industry partner Nicander Ltd, presented at the *Student Success Through Partnership* conference) established the 3D-vision foundation I bring to that work.

## Tech
Python · PyTorch / TensorFlow · PointNet · Open3D · NumPy · LiDAR / point-cloud processing.

Author: **Keyhan Azarjoo**.
