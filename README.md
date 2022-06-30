milk_cls
==============================

Classification of milk based on dried drop images of milks

Image processing tools:

* OpenCV
* Scikit-image
* Pillow

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" 
href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

Pancake & Doughnut morphologies of dark blue milk at lower temperatures
==============================

Feature extraction
------------
* Find one circle in pancake and  two circles (outside and inside) doughnut images by using the Hough circle transform.
  * Notebooks
    * 1.0-bl-initial-data-exploration: read and display the images given.
    * 2.0-bl-detect-circle: detect circles from images
    * 3.0-Bruce-Inner-cycle and 04-bruce_inner_cycle: try to detect the inner circle from Doughnut images.
    * 5.0-Bruce-extract-circles and 5.0-Bruce-extract-circles-auto: extract the pixels from the circled images.
    * 6.0_auto_two_circles: extract pixels from two circled images
    * 7.0-original-lbp-histogram: utilize LBP technology to extract image features.
    * 7.0_auto_outside_circle_lbp_histogram: LBP to extracted features from circled images 

Classification
------------
* Notebooks
  * 9.0-image-classification: KNN and GradientBoosting are to classify PC and DN. The notebook calls the pipeline function:
    * multiple_cls_test: from src.models.train_model import multiple_cls_test



Dark blue, mixture(dbm_gm) and green_trim in bottom or side lights
==============================

Feature extraction
------------
* Given the center and extract five rings from images.
* Notebooks:
  * 11-new-images-lbp: utilize LBP methods to extract features from different 5 rings which centers are same. The radis of five rings increase by 100 pixels from innter to outside.
Classification
  * 12-new-images-lbp-classification: three classifiers are to identify three catagories by lbp features of each ring, merged rings.
------------
* Notebook: 17-combining-rings-lbp-classification-merge-central-ring