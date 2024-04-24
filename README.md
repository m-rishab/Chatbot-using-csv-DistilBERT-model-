# Children Details Chatbot

The Children Details Chatbot is a natural language processing (NLP) project that allows users to query details about children from a dataset. The chatbot is powered by a pretrained DistilBERT model for question answering and provides information such as a child's name, age, gender, grade, favorite hobby, favorite subject, favorite food, and family activity.

## Overview

This project consists of the following components:

- **Dataset**: The chatbot uses a dataset named `Children_data.csv` containing information about children.
- **Model**: A DistilBERT model is fine-tuned for sequence classification to predict children's attributes.
- **Streamlit App**: An interactive Streamlit app provides the user interface for the chatbot.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/children-details-chatbot.git
   
## Install the required Python libraries:

####'pip install -r requirements.txt'

## Run the Streamlit app:

--streamlit run app.py
--The app will open in your default web browser.
--Enter a question about a child's details in the text input box.
--Click the "Get Details" button to get information and an answer.

### Supported Queries
---Users can ask questions about a child's:

Name,
Age,
Gender,
Grade,
Health Condition,
Favorite Hobby,
Favorite Subject,
Favorite Food,
Family Activity

### Code Structure:
- **app.py:** Main Streamlit app for the chatbot interface.
- **data/Children_data.csv:** Dataset containing children's details.
- **models/:** Directory containing the fine-tuned DistilBERT model.
- **README.md:** Detailed information about the project.
- **requirements.txt:** List of Python dependencies.

### Demo
<video width="500" controls>
  <source src="https://github.com/m-rishab/Chatbot-using-csv-DistilBERT-model-/assets/113618652/37f9aebd-b611-4206-ab1d-1f5870628d22" type="video/mp4">
  Your browser does not support the video tag.
</video>

