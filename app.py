import streamlit as st
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering

# Load the dataset
df = pd.read_csv('data/Children_data.csv')

# Load pretrained DistilBERT model and tokenizer
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")

# Function to predict child details
def predict_child_details(query):
    # Extract the name from the query
    name = query.split()[-1].strip("?")
    
    # Search for the name in the dataset
    row = df[df['Name'] == name]
    
    if len(row) == 0:
        return {"error": f"Child with the name '{name}' not found in the dataset"}
    
    details = {}
    details['Name'] = row['Name'].iloc[0]
    details['Age'] = row['Age'].iloc[0]
    details['Gender'] = row['Gender'].iloc[0]
    details['Grade'] = row['Grade'].iloc[0]
    details['Health Condition'] = row['Health Condition'].iloc[0]
    details['Favorite Hobby'] = row['Favorite Hobby'].iloc[0]
    details['Favorite Subject'] = row['Favorite Subject'].iloc[0]
    details['Favorite Food'] = row['Favorite Food'].iloc[0]
    details['Family Activity'] = row['Family Activity'].iloc[0]
    
    return details

# Main Streamlit app
def main():
    st.title("Children Details Chatbot")
    st.write("This chatbot can provide details about children from the dataset.")

    # User input for question
    query = st.text_input("Ask a question (e.g., What is the age of Aarav?)")

    if st.button("Get Details"):
        if not query:
            st.warning("Sorry, you did not provide any query.")
        else:
            # Predict child details
            details = predict_child_details(query)

            if "error" in details:
                st.error(details["error"])
            else:
                st.success("Here are the details:")
                st.write("Name:", details['Name'])
                st.write("Age:", details['Age'])
                st.write("Gender:", details['Gender'])
                st.write("Grade:", details['Grade'])
                st.write("Health Condition:", details['Health Condition'])
                st.write("Favorite Hobby:", details['Favorite Hobby'])
                st.write("Favorite Subject:", details['Favorite Subject'])
                st.write("Favorite Food:", details['Favorite Food'])
                st.write("Family Activity:", details['Family Activity'])

                # Question answering with pretrained DistilBERT
                context = f"{details['Name']} is a {details['Age']}-year-old {details['Gender']} in grade {details['Grade']}. " \
                          f"Their favorite hobby is {details['Favorite Hobby']}, and they love studying {details['Favorite Subject']}. " \
                          f"They enjoy eating {details['Favorite Food']} and spending time with family, especially {details['Family Activity']}."

                question = query
                inputs = tokenizer.encode_plus(question, context, return_tensors="pt", add_special_tokens=True)
                input_ids = inputs["input_ids"].tolist()[0]

                # Get the answer
                answer = model(**inputs)
                answer_start = torch.argmax(answer.start_logits)
                answer_end = torch.argmax(answer.end_logits)
                answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end+1]))

                if answer.strip() == '[CLS]':
                    st.warning("Sorry, I couldn't find an answer to that question.")
                else:
                    st.info("Answer to your question:")
                    st.write(answer)

# Run the app
if __name__ == "__main__":
    main()
