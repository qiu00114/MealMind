# MealMind

MealMind is a **context-aware personalized meal recommendation system** that helps users quickly decide what to eat. Instead of relying only on nutrition scores or static user preferences, MealMind integrates real-world contextual factors such as **time of day**, **season**, and **weekday patterns** to generate more relevant suggestions. By combining **content-based filtering**, **collaborative filtering**, and structured contextual features, the system learns from user interactions and provides recommendations that feel intuitive, adaptive, and personalized over time.

---

## Features

* Context-aware recommendation (time, season, meal-type logic)
* Hybrid system: content-based + collaborative filtering
* Large-scale recipe and interaction data support
* Evaluation metrics: Precision@K, Slot Hit Rate, Diversity Index

---

## Project Structure

```
.
├── data/
│   ├── raw/                # Original Kaggle CSV files
│   └── processed/          # Cleaned / merged data files
├── notebooks/
│   ├── 01_load_data.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── data_utils.py
│   ├── models.py
│   └── evaluation.py
├── README.md
└── requirements.txt
```

---

## Setup

1. Clone the repository:

```
git clone https://github.com/your-username/MealMind.git
cd MealMind
```

2. (Optional) Create a virtual environment:

```
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Download the Food.com dataset and place CSV files into `data/raw/`.

---

## Usage

* Use the notebooks to load and clean data, train models, and evaluate results.
* Optionally integrate core logic into a simple interface for one-click meal generation.

---

## License

MIT License.
