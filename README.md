# MNIST Handwritten Digit Classifier

A simple deep learning web app that recognizes handwritten digits (0–9) drawn on a canvas, powered by FastAPI, PyTorch, and React (Vite).

This project was inspired by the paper on CNN-based architectures for digit recognition:  
[MNIST Handwritten Digit Recognition with Different CNN Architectures (CloudFront PDF)](https://dif7uuh3zqcps.cloudfront.net/wp-content/uploads/sites/11/2021/01/17192613/MNIST-Handwritten-Digit-Recognition-with-Different-CNN-Architectures.pdf)

---

## Overview

This project combines a trained Convolutional Neural Network (CNN) built with PyTorch, a FastAPI backend, and a modern React frontend interface.

- Draw any digit (0–9) in the browser canvas.
- The frontend captures the image and sends it to the FastAPI backend.
- The backend loads a pretrained model (`mnist_cnn.pt`) to predict the digit.
- The result is displayed instantly in your browser.

---

## Stack

**Backend:** FastAPI, PyTorch, Uvicorn  
**Frontend:** React + Vite  
**Deployment:** Render (Dockerized API + Static Web)  
**Model:** CNN trained on the MNIST dataset  

---
