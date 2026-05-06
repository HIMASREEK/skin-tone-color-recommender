# 👗✨ Skin Tone-Based Clothing Color Recommendation API

[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/ChromaFit-Project/backend_machine_learning)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/github/license/ChromaFit-Project/backend_machine_learning)](https://github.com/ChromaFit-Project/backend_machine_learning/blob/main/LICENSE)
[![Issues](https://img.shields.io/github/issues/ChromaFit-Project/backend_machine_learning.svg)](https://github.com/ChromaFit-Project/backend_machine_learning/issues)
[![Forks](https://img.shields.io/github/forks/ChromaFit-Project/backend_machine_learning.svg)](https://github.com/ChromaFit-Project/backend_machine_learning/network/members)
[![Stars](https://img.shields.io/github/stars/ChromaFit-Project/backend_machine_learning.svg)](https://github.com/ChromaFit-Project/backend_machine_learning/stargazers)
[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ChromaFit-Project/backend_machine_learning/pulls)
[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ChromaFit-Project/backend_machine_learning)

---

## 🧠 Overview

This project uses machine learning to **classify skin tones** and recommend the **best clothing color palettes** for each tone using HEX codes. It is particularly helpful for:

- Fashion stylists 👗  
- E-commerce clothing stores 🛍️  
- Individuals seeking personalized wardrobe insights 🧥  

## 🏗️ Project Structure

```bash
.
├── data/
│   ├── skin_shades_india.csv           # 56 skin tones with HEX codes
│   └── fashion_color_palette.csv       # Clothing colors by tone
├── output/
│   └── final_recommendations.csv       # Final recommendation output
├── skin_tone_mapping.py                # Main ML logic
├── requirements.txt                    # Python dependencies
├── README.md                           # You’re reading it!
````

## 🔍 Methodology

### 1. Data Preprocessing

* Convert HEX → RGB
* Normalize and clean color data

### 2. Skin Tone Classification

* Use `KMeans` clustering to group similar tones
* Assign tones to closest clothing palette category

### 3. Color Recommendation

* Merge skin tone categories with fashion color palettes
* Recommend 2–3 matching clothing colors per skin tone

---

## 📦 Dependencies

* Python 3.8+
* pandas
* numpy
* scikit-learn
* matplotlib (optional, for visualizations)

Install with:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
python skin_tone_mapping.py
```

After execution, the `final_recommendations.csv` will be generated in the `/output` folder.

### ✅ Sample Output

| Skin Shade ID | Skin HEX Code | Assigned Category | Recommended Colors       | Color HEX Codes      |
| ------------- | ------------- | ----------------- | ------------------------ | -------------------- |
| Shade 1       | `#F5E0D8`     | Cool Tones        | Navy Blue, Emerald Green | `#000080`, `#50C878` |
| Shade 2       | `#C68642`     | Warm Tones        | Coral, Mustard Yellow    | `#FF7F50`, `#FFDB58` |

---

## 💡 API Integration (Coming Soon)

We are building a Flask/FastAPI-based interface to allow:

* Skin tone HEX input via API
* Receive recommended color palettes as JSON
* Real-time predictions via model inference

Stay tuned! 🚧

---

## 🌱 Future Enhancements

* 🔍 Use CNNs for more robust skin tone detection from images
* 🌎 Expand datasets globally beyond India
* 📱 Deploy as a web/mobile app
* 📊 Integrate with e-commerce sites for live recommendations
* 🧠 Add feedback learning loop for smarter suggestions

---



## ⭐ If you like this project, don't forget to ⭐ the repo!`
