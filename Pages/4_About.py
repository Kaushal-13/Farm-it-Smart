import streamlit as st


def about_us():
    st.title('About Our Farming Solutions App')

    st.header('Our Mission')
    st.write("""
    Our mission is to bridge the gap between traditional farming practices and cutting-edge technology. 
    We aim to provide farmers with intuitive tools that harness the power of data to enhance crop yield, 
    minimize risks, and optimize resource utilization.
    """)

    st.header('What We Offer')

    st.subheader('Image Classification')
    st.write("""
    Our Image Classification feature allows farmers to upload images of crop leaves and receive immediate 
    analysis on their health status. Utilizing advanced machine learning algorithms, 
    we provide real-time assessments to identify potential issues and guide farmers in crop management decisions.
    """)

    st.subheader('Crop Prediction')
    st.write("""
    Through our Crop Prediction tool, farmers can input location-specific details and basic soil characteristics 
    to receive tailored recommendations for suitable crops. Our predictive models consider environmental factors 
    to suggest the most viable crops, enhancing farming success rates.
    """)

    st.header('Our Commitment')
    st.write("""
    We prioritize user-centric design and reliability, ensuring our solutions are user-friendly, 
    accurate, and continually updated to meet the evolving needs of the farming community.
    """)


# Display the About Us page when the app runs
about_us()
