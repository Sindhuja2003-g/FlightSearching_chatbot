import streamlit as st
from flight_api import search_flights
from predict_model import predict_class


st.title("JetFinder")
st.write("Hello! I'm here to assist you. Type your queries...")

chat_container = st.container()

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Flight session data
if "flight_data" not in st.session_state:
    st.session_state.flight_data = {
        "source": None,
        "destination": None,
        "date": None,
        "adults": None,
    }


def display_chat():  #display chat in order
    for user_msg, bot_msg in st.session_state.history:
        chat_container.markdown(
            f'<div style="background-color:blue; color:white; padding:10px; border-radius:10px; margin-bottom:5px; width:fit-content; border: 1px solid #ccc; float:right;">{user_msg}</div>',
            unsafe_allow_html=True,
        )
        chat_container.markdown(
            f'<div style="background-color:gray; color:black; padding:10px; border-radius:10px; margin-bottom:5px; width:fit-content;">Bot: {bot_msg}</div>',
            unsafe_allow_html=True,
        )


user_input = st.text_input("You:", key="input")  

if user_input:

    tag = predict_class(user_input)  #prediction
    
    if tag == "greeting":
        response = "Hello! How can I assist you today?"

    elif tag == "flight_query":
       
        st.session_state.flight_data["source"] = st.text_input("Source:", value=st.session_state.flight_data["source"], key="source")
        st.session_state.flight_data["destination"] = st.text_input("Destination:", value=st.session_state.flight_data["destination"], key="destination")
        st.session_state.flight_data["date"] = st.text_input("Travel Date (YYYY-MM-DD):", value=st.session_state.flight_data["date"], key="date")
        st.session_state.flight_data["adults"] = st.number_input("Number of adults:", min_value=1, value=st.session_state.flight_data["adults"], key="adults")

        
        if not st.session_state.flight_data["source"] or not st.session_state.flight_data["destination"] or not st.session_state.flight_data["date"] or not st.session_state.flight_data["adults"]:
            response = None  #wait to provide details
        else:
            try:
                result = search_flights(
                    st.session_state.flight_data["source"],
                    st.session_state.flight_data["destination"],
                    st.session_state.flight_data["date"],
                    int(st.session_state.flight_data["adults"]),
                )

                
                formatted_response = "<div style='background-color: white; color: black; padding: 20px; border-radius: 10px; font-family: Arial, sans-serif;'>"
                formatted_response += "<h3>Flight Details</h3>"

                if len(result) > 0:
                    for flight in result:
                        formatted_response += f"<p>{flight[0]}</p>"  
                        formatted_response += f"<p>{flight[1]}</p>"  # Departure
                        formatted_response += f"<p>{flight[2]}</p>"  # Arrival
                        formatted_response += f"<p>{flight[3]}</p>"  # Aircraft
                        formatted_response += f"<p>{flight[4]}</p>"  # price
                        formatted_response += "<hr style='border: 1px solid black;'>"
                else:
                    formatted_response += "<p>No flights found. Please check your input details.</p>"

                formatted_response += "</div>"
                response = formatted_response 

            except Exception as e:
                response = (
                    "I encountered an error while searching for flights."
                    "Please double-check your input details."
                )

            st.session_state.flight_data = {"source": None, "destination": None, "date": None, "adults": None}  #resetting

    elif tag == "goodbye":
        response = "Goodbye! Have a great day!"

    else:
        response = "I didn't understand you."


    if response:
        st.session_state.history.append((user_input, response))

display_chat()
