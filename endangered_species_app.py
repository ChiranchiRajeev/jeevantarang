import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from io import BytesIO
from fpdf import FPDF
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import tempfile
from folium import plugins

# Load dataset from Excel file
def load_data(file_path):
    data = pd.read_excel(r"C:\Users\dhoni\Videos\jt3\Endangered_Species_Dataset.xlsx")
    return data

def preprocess_data(data):
    label_encoders = {}
    categorical_columns = ['Region', 'Habitat_Type']

    # Encode categorical columns
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Scale numerical columns
    scaler = StandardScaler()
    numerical_columns = [
        'Current_Population', 'Population_Decline_Rate (%)', 'Average_Temperature (°C)',
        'Air_Quality_Index', 'Noise_Level (dB)', 'Protected_Areas (%)',
        'Migration_Distance (km)', 'Climate_Change_Risk (%)', 'Fragmentation_Risk (%)'
    ]
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # **Exclude 'Image_URL' from the feature set**
    X = data.drop(columns=['Species', 'Extinction_Risk (%)', 'Image_URL', 'Region', 'Habitat_Type'], errors='ignore')
    y = data['Extinction_Risk (%)']
    return X, y, scaler, label_encoders, numerical_columns, categorical_columns



# Geospatial visualization with image URL

def plot_species_on_map(data):
    m = folium.Map(location=[20, 0], zoom_start=2)
    
    for _, row in data.iterrows():
        # Check if 'Image_URL' exists for the species
        image_url = row.get("Image_URL", None)
        popup_content = f"<b>{row['Species']}</b><br>Extinction Risk: {row['Extinction_Risk (%)']}%<br>Region: {row['Region']}<br>Habitat: {row['Habitat_Type']}"
        
        # If there's an image URL, add it to the popup
        if image_url:
            popup_content += f"<br><img src='{image_url}' width='250px' height='150px'>"
            
            # Create a custom DivIcon to use the image as the marker
            icon = folium.DivIcon(
                icon_size=(40, 40),  # Size of the icon
                icon_anchor=(20, 40),  # Position of the icon relative to the marker
                html=f'<img src="{image_url}" width="30" height="30" style="border-radius: 50%;">'
            )

            # Add the marker with the image as the icon
            folium.Marker(
                location=[row.get("Latitude", 0), row.get("Longitude", 0)],
                popup=folium.Popup(popup_content, max_width=300),
                icon=icon  # Use the custom DivIcon with the image
            ).add_to(m)

    st_folium(m, width=800, height=500)





def generate_report_with_graph(data):
    # Generate a bar plot for Species vs Extinction Risk
    plt.figure(figsize=(10, 6))
    plt.bar(data['Species'], data['Extinction_Risk (%)'], color='skyblue')
    plt.xlabel('Species')
    plt.ylabel('Extinction Risk (%)')
    plt.title('Extinction Risk by Species')
    plt.xticks(rotation=90)

    # Save the plot to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        plt.tight_layout()  # Adjust layout to prevent clipping
        plt.savefig(tmpfile.name, format='png')
        plot_path = tmpfile.name
    plt.close()

    # Create the PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add a title
    pdf.cell(200, 10, txt="Species Extinction Risk Report", ln=True, align='C')

    # Add species details and extinction risk
    pdf.set_font("Arial", size=10)
    for _, row in data.iterrows():
        pdf.cell(200, 10, txt=f"Species: {row['Species']}", ln=True)
        pdf.cell(200, 10, txt=f"Extinction Risk: {row['Extinction_Risk (%)']}%", ln=True)
        pdf.cell(200, 5, txt="", ln=True)  # Blank line for spacing

    # Add the graph to the PDF
    pdf.add_page()
    pdf.cell(200, 10, txt="Graph: Extinction Risk by Species", ln=True, align='C')
    pdf.image(plot_path, x=10, y=30, w=180)  # Adjust x, y, and width as needed

    # Save PDF to a buffer for download
    pdf_buffer = BytesIO()
    pdf_output = pdf.output(dest='S').encode('latin1')  # Capture the output as a string
    pdf_buffer.write(pdf_output)  # Write to the buffer
    pdf_buffer.seek(0)  # Go to the beginning of the buffer

    # Provide the buffer as a downloadable file in Streamlit
    st.sidebar.download_button("Download PDF Report", pdf_buffer, file_name="species_report.pdf", mime="application/pdf")


# Visualizations for parameter comparison
def parameter_comparison(data, parameter):
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Species", y=parameter, data=data)
    plt.title(f'Comparison of {parameter} Across Species')
    plt.xticks(rotation=90)
    st.pyplot(plt)

# Streamlit app

# Pre-load the Excel file (update the file path as needed)
data = pd.read_excel(r"C:\Users\dhoni\Videos\jt3\Endangered_Species_Dataset.xlsx")  # Change this to your file path

def main():
    st.set_page_config(page_title="Species Extinction Risk", layout="wide")

    st.title("JEEVANTARANG ⧖")
    st.markdown("Explore species extinction risks with interactive data visualizations and insights.")

    # Dataset preview
    st.subheader("Dataset Preview")
    st.write(data.head())

    # Sidebar for variable selection
    st.sidebar.header("Model Configuration")
    dependent_variable = st.sidebar.selectbox("Select Dependent Variable (Y):", data.columns)
    independent_variables = st.sidebar.multiselect(
        "Select Independent Variables (X):", [col for col in data.columns if col != dependent_variable]
    )

    if dependent_variable and independent_variables:
        # Preprocess the data
        X, y, scaler, label_encoders, numerical_columns, categorical_columns = preprocess_data(data)
        X = X[independent_variables]
        y = data[dependent_variable]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Linear Regression Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * ((len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1))

        # Coefficients
        coefficients = pd.DataFrame({
            "Feature": independent_variables,
            "Coefficient": model.coef_
        })

        # Display metrics and insights
        st.sidebar.subheader("Model Insights")
        st.sidebar.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.sidebar.write(f"**R-squared:** {r2:.4f}")
        st.sidebar.write(f"**Adjusted R-squared:** {adj_r2:.4f}")

        if r2 > 0.7:
            st.sidebar.write("🟢: The model explains a significant portion of the variance in the dependent variable.")
        elif r2 > 0.4:
            st.sidebar.write("🟡: The model explains a moderate portion of the variance. Consider adding more features or refining the data.")
        else:
            st.sidebar.write("🔴: The model has low explanatory power. Significant improvements are needed.")

        if mse < 1:
            st.sidebar.write("🟢: The model's predictions are very close to the actual values on average.")
        else:
            st.sidebar.write("🔴: The model's predictions have a higher average error. Further tuning might help.")

        # Coefficients table
        st.subheader("Feature Coefficients")
        st.write(coefficients)

        # Visualizations
        st.subheader("Visualizations")
        if len(independent_variables) == 1:
            # Scatterplot for single variable
            plt.figure(figsize=(8, 6))
            plt.scatter(X_test[independent_variables[0]], y_test, color='blue', label='Actual')
            plt.plot(X_test[independent_variables[0]], y_pred, color='red', label='Predicted')
            plt.title(f"Regression Line for {independent_variables[0]}")
            plt.xlabel(independent_variables[0])
            plt.ylabel(dependent_variable)
            plt.legend()
            st.pyplot(plt)
        else:
            # Residual plot for multiple variables
            residuals = y_test - y_pred
            plt.figure(figsize=(8, 6))
            plt.scatter(y_pred, residuals, color='purple')
            plt.axhline(y=0, color='red', linestyle='--')
            plt.title("Residual Plot")
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")
            st.pyplot(plt)

    species = data["Species"].unique()
    selected_species = st.sidebar.selectbox("Select Species", species)

    if selected_species:
        # Filter data for the selected species
        selected_row = data[data["Species"] == selected_species].iloc[0]

        # Display species details with image beside it
        st.header(f"Species: {selected_row['Species']}")

        # Two-column layout
        col1, col2 = st.columns([1, 3])

        # Column 1 (for image)
        image_url = selected_row.get("Image_URL", None)
        if pd.notnull(image_url):
            col1.markdown(
                f"""
                <div style="text-align: center;">
                    <img src="{image_url}" style="width: 250px; height: auto; border-radius: 15px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);"/>
                    <p style="font-size: 14px; font-style: italic;">Image of {selected_row['Species']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            col1.warning("No image available for this species.")

        # Column 2 (for information)
        col2.write(f"**Region**: {selected_row['Region']}")
        col2.write(f"**Habitat Type**: {selected_row['Habitat_Type']}")
        col2.write(f"**Current Population**: {selected_row['Current_Population']}")
        col2.write(f"**Population Decline Rate**: {selected_row['Population_Decline_Rate (%)']}%")
        col2.write(f"**Climate Change Risk**: {selected_row['Climate_Change_Risk (%)']}%")
        col2.write(f"**Fragmentation Risk**: {selected_row['Fragmentation_Risk (%)']}%")
        col2.write(f"**Extinction Risk**: {selected_row['Extinction_Risk (%)']}%")

        # Risk warnings
        if selected_row['Climate_Change_Risk (%)'] > 80 or selected_row['Fragmentation_Risk (%)'] > 70:
            st.warning("High extinction risk due to climate change or habitat fragmentation.")
        elif selected_row['Current_Population'] < 5000:
            st.warning("Critical risk due to low population levels.")
        else:
            st.success("Moderate risk, but monitoring is advised.")

    comparison_parameter = st.sidebar.selectbox(
        "Select Column for Comparison",
        [
            'Current_Population', 'Population_Decline_Rate (%)', 'Average_Temperature (°C)',
            'Air_Quality_Index', 'Noise_Level (dB)', 'Protected_Areas (%)', 'Migration_Distance (km)',
            'Climate_Change_Risk (%)', 'Fragmentation_Risk (%)', 'Extinction_Risk (%)'
        ]
    )

    if comparison_parameter:
        st.header(f"Comparing {comparison_parameter} with Species")
        parameter_comparison(data, comparison_parameter)

    st.header("Geospatial Visualization")
    plot_species_on_map(data)

    generate_report_with_graph(data)

if __name__ == "__main__":
    main()
