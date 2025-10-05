# ExoVision AI — README

**ExoVision AI** is a hackathon-ready MVP that classifies Kepler Objects of Interest (KOI) using a pre-trained LightGBM classifier and optionally fine-tunes the model in-memory with user-provided labeled data. The project pairs a polished front-end (HTML UI) with a simple Flask backend (`/api/process`) that accepts `.xlsx` uploads and returns JSON results.

This README covers: setup, usage, file formats, API contract, fine-tuning behavior, deployment notes, security, troubleshooting, and suggestions for improvement.

---

## Table of contents
- [What it does](#what-it-does)  
- [Project structure](#project-structure)  
- [Requirements](#requirements)  
- [Quickstart (development)](#quickstart-development)  
- [How the API works](#how-the-api-works)  
- [Data / Excel format](#data--excel-format)  
- [Example requests (curl & JS)](#example-requests-curl--js)  
- [Fine-tuning details](#fine-tuning-details)  
- [Frontend integration](#frontend-integration)  
- [Security & privacy considerations](#security--privacy-considerations)  
- [Performance tuning & tips](#performance-tuning--tips)  
- [Troubleshooting & FAQ](#troubleshooting--faq)  
- [Suggested improvements](#suggested-improvements)  
- [Credits & License](#credits--license)

---

## What it does
- Accepts an Excel file (`.xlsx`) containing KOI observations.
- Two modes:
  - **predict**: Use a pre-trained LightGBM (`model.pkl`) to return classification probabilities and labels.
  - **retrain**: Fine-tune the LightGBM model in-memory using uploaded labeled data and return updated predictions on the provided data.
- Returns structured JSON with predictions, probabilities, and used features.
- Fine-tuned model is kept **in memory** for the running Flask process (not persisted by default).

---

## Project structure
