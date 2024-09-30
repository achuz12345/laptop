import pandas as pd
import joblib
import streamlit as st

# Define possible values for each parameter
brands = ['hp', 'lenovo', 'acer', 'apple', 'asus', 'mi']
rams = ['2gb', '4gb', '8gb', '16gb', '32gb']
screen_sizes = [13, 14, 15, 16, 17]
processors = ['ryzen 3', 'i3', 'ryzen 5', 'i5', 'ryzen 7', 'i7', 'ryzen 9', 'i9']
gpus = ['gtx 1050', 'gtx 1650', 'rtx 2060', 'rtx 3060', 'rtx 3070', 'rtx 3080']

# Load the trained model
model_path = 'laptop_price_model.pkl'
try:
    model = joblib.load(model_path)
except (FileNotFoundError, ValueError) as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the dataset for additional details
df = pd.read_csv('laptop_price_prediction_extended.csv')

# Streamlit app
st.title('Laptop Price Prediction and Recommendation')

# User inputs
brand = st.selectbox('Select Brand', brands)
ram = st.selectbox('Select RAM', rams)
screen_size = st.selectbox('Select Screen Size', screen_sizes)
processor = st.selectbox('Select Processor', processors)
gpu = st.selectbox('Select GPU', gpus)

# Create input dataframe
input_data = pd.DataFrame([[brand, ram, screen_size, processor, gpu]],
                          columns=['brand', 'ram', 'screen_size', 'processor', 'gpu'])

# Predict price
if st.button('Predict Price'):
    try:
        price = model.predict(input_data)[0]
        st.markdown(f"## Predicted Price: ₹{price:,.2f}")

        # Filter the dataset to find similar laptops
        similar_laptops = df[
            (df['brand'] == brand) &
            (df['ram'] == ram) &
            (df['screen_size'] == screen_size) &
            (df['processor'] == processor) &
            (df['gpu'] == gpu)
        ]

        if not similar_laptops.empty:
            st.write('Details of Recommended Laptops:')
            for _, laptop in similar_laptops.iterrows():
                st.markdown(f"### **{laptop['brand']}**")
                st.write(f"**Description:** {laptop['description']}")
                st.write(f"**Operating System:** {laptop['os']}")
                st.write(f"**RAM:** {laptop['ram']}")
                st.write(f"**Screen Size:** {laptop['screen_size']} inches")
                st.write(f"**Processor:** {laptop['processor']}")
                st.write(f"**GPU:** {laptop['gpu']}")
                st.write(f"**Warranty:** {laptop['warranty']}")
                st.write(f"**Price:** ₹{laptop['price']:,.2f}")
                st.write("---")
        else:
            st.write('No similar laptops found in the dataset.')
    except Exception as e:
        st.error(f"Error making prediction: {e}")
