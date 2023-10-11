# TextRecognition: Handwritten Text Recognition using TensorFlow
A rich collection of over 400,000 handwritten names, meticulously categorized into training, testing, and validation sets. Sourced from charity projects, this dataset aims to propel research in converting handwritten text into digital format, accounting for the vast spectrum of individual writing styles.


This repository contains a deep learning model to recognize handwritten texts using TensorFlow and Keras. The model architecture leverages a combination of CNNs for feature extraction and RNNs for sequence modeling, followed by CTC loss for sequence alignment.

# Overview

The dataset you're navigating is a collection of over 400,000 handwritten names, all generously amassed through charity-driven endeavors.

In the realm of digitization, Character Recognition plays a pivotal role. It leverages cutting-edge image processing technologies to metamorphose characters from scanned documents into their digital counterparts. While machines have become adept at interpreting machine-printed fonts, handwritten characters, marked by their unique individualistic styles, remain a challenging frontier.

This dataset is a testament to that variety. It's an assortment of 206,799 first names complemented by 207,024 surnames. To facilitate the developmental lifecycle, the data has been meticulously divided into distinct sets: training (331,059 entries), testing (41,382 entries), and validation (41,382 entries).

# Content

At the heart of this repository are myriad images, each bearing a handwritten name. In the dataset directory, you'll discover these transcribed images neatly categorized into test, training, and validation sets.

To provide a coherent structure, the image labels adhere to a specific naming convention. This systematic approach ensures that enthusiasts and researchers can seamlessly integrate their own data. Here's a glimpse of the naming format:

# Image	# URL				
D2M	15	0010079F	0002	1	first name.jpg
D2M	15	0010079F	0002	1	surname.jpg
...
D2M	15	0010079F	0006	5	surname.jpg

# Inspiration

This dataset was conceived with a singular vision: to propel research in classifying handwritten text. The ultimate aspiration is to harness technology and methodologies to transpose handwritten characters into digital text. It's an invitation for enthusiasts, researchers, and professionals to experiment, innovate, and discover pathways to bridge the analog-digital divide in the world of text.

## Project Structure

- **App.py**: Main script to train and evaluate the handwritten text recognition model.
- **written_name_train_v2.csv**: Training dataset labels.
- **written_name_validation_v2.csv**: Validation dataset labels.
- **train_v2/**: Folder containing training images.
- **validation_v2/**: Folder containing validation images.

## Model Architecture

1. **CNN Layers**: Extracts feature maps from the input images.
2. **RNN Layers**: Models the sequence of the features extracted by the CNN.
3. **CTC Loss**: Aligns the predicted sequences with the ground truth.

## Getting Started

### Prerequisites

Ensure you have the following dependencies installed. You can also find them in the `requirements.txt` file:
tensorflow>=2.6
pandas
numpy
opencv-python


To install them, use: pip install -r requirements.txt


### Training the Model

Navigate to the project's root directory and run:

python3 App.py

## Results

The model is saved as `htr_model_ctc.h5` after training. You can further fine-tune or evaluate this saved model.

## Contributing

Please read [CONTRIBUTING.md](https://github.com/yourusername/TextRecognition/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md]file for details.
Copyright (c) 2023 Salman MInshawi.

## Acknowledgments

- OpenAI for guidance and insights.
- TensorFlow and Keras documentation.
