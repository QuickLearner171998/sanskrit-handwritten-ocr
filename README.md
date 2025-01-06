# Data Preparation
`
1. Sample and Convert tif to png and resize - `sanskrit-handwritten-ocr/training_data_prep/images_sampling.py`
2. Dump pcikle files from Google-Cloud_vision - `sanskrit-handwritten-ocr/training_data_prep/google_ocr_response_dump.py`
3. Create Preannotations CVAT format - `sanskrit-handwritten-ocr/training_data_prep/create_preannotations_cvat.py`
4. Upload annotations to CVAT and correct the annotations
5. Split the data and annotations - `sanskrit-handwritten-ocr/training_data_prep/ppocr_data_prep.py`
6. Reformat to PPOCR or EasyOCR- `sanskrit-handwritten-ocr/training_data_prep/ppocr_data_prep.py`
