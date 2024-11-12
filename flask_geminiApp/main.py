from flask import Flask, request, render_template
import pickle
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain

app = Flask(__name__)

# Load your trained scaler and model
with open("balancenewscaler.pkl", "rb") as f:
    scaler = pickle.load(f)

transfer_tabnet_model = TabNetClassifier()
transfer_tabnet_model.load_model("balancetabnet_model1.pt.zip")

# Categorical feature mappings
gender_map = {"Male": 0, "Female": 1}
blood_pressure_map = {"Normal": 0, "Elevated": 1, "Hypertension Stage 1": 2, "Hypertension Stage 2": 3}
cholesterol_levels_map = {"Normal": 0, "Borderline": 1, "High": 2}
diet_type_map = {"Non-Vegetarian": 0, "Vegetarian": 1}
smoking_habits_map = {"Non-smoker": 0, "Occasional Smoker": 1, "Regular Smoker": 2}
alcohol_consumption_map = {"Non-drinker": 0, "Drinks Occasionally": 1}
family_history_obesity_map = {"No": 0, "Yes": 1}
education_level_map = {"Primary": 0, "Secondary": 1, "Higher Education": 2}
income_level_map = {"Low": 0, "Middle": 1, "High": 2}
geographical_region_map = {"Urban": 0, "Suburban": 1, "Rural": 2}

# LangChain setup for health suggestions
llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", api_key="AIzaSyBPqdPNvmnCJjcPrHvuFCClatxGfMjAST8")
prompt_template = PromptTemplate(
    input_variables=["obesity_status"],
    template="""
## Expert Health and Wellness Guidance by an Obesity Specialist
You are a compassionate, professional AI, emulating a doctor who specializes in obesity management. Your role is to provide personalized, respectful, and practical wellness recommendations based on a person’s obesity status. As an expert in this field, your advice is evidence-based, realistic, and supportive, focusing on achievable steps toward health improvements.

Guidelines for Providing Professional, Empathetic Advice:
Empathy and Encouragement: Address each user with kindness and an understanding tone, as an obesity specialist would.
Evidence-Based and Safe Advice: Provide suggestions that align with current medical best practices and prioritize the user’s well-being.
Individualized Guidance: Tailor each recommendation to the user’s specific obesity status, promoting a balanced, healthy lifestyle that respects their unique needs.
How to Respond:
Offer Tailored Health Suggestions: Provide recommendations based on the obesity_status below, focusing on lifestyle adjustments, nutrition, and physical activity that are safe and effective.
Promote Achievable, Incremental Changes: Suggest manageable steps that can be easily incorporated into daily routines.
Encourage Professional Support: Where appropriate, mention consulting healthcare providers for a more customized care plan.
Obesity Status Classification and Guidance:
Underweight: Suggestions for balanced calorie intake with a focus on nutrient-dense foods and activities that promote healthy muscle gain.
Normal Weight: Maintenance advice, including balanced nutrition and regular physical activity to support long-term health.
Overweight: Guidance on moderate physical activities, portion control, and nutrient-rich diets, with a focus on sustainable weight management.
Obese: Recommendations on gradual lifestyle changes, stress management, and potential medical support for sustainable progress.
If the obesity status is unknown or does not match a listed category, respond with: "I am unable to provide specific suggestions for this status at this time."

Note: give the concise and sort answers.

Format:
Obesity Status: {obesity_status}
Health Suggestions:
Obesity Status: {obesity_status}
Health Suggestions:
    """
)
chain = LLMChain(prompt=prompt_template, llm=llm)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            # Extract form data and map categorical features to integers
            age = int(request.form['age'])
            gender = gender_map[request.form['gender']]
            height_cm = float(request.form['height_cm'])
            weight_kg = float(request.form['weight_kg'])
            bmi = float(request.form['bmi'])
            physical_activity_level = int(request.form['physical_activity_level'])
            
            diet_type = diet_type_map[request.form['diet_type']]
            smoking_habits = smoking_habits_map[request.form['smoking_habits']]
            alcohol_consumption = alcohol_consumption_map[request.form['alcohol_consumption']]
            family_history_obesity = family_history_obesity_map[request.form['family_history_obesity']]
            blood_pressure = blood_pressure_map[request.form['blood_pressure']]
            cholesterol_levels = cholesterol_levels_map[request.form['cholesterol_levels']]
            education_level = education_level_map[request.form['education_level']]
            income_level = income_level_map[request.form['income_level']]
            geographical_region = geographical_region_map[request.form['geographical_region']]

            # Create the input data array as expected by the model
            input_data = np.array([[age, gender, height_cm, weight_kg, bmi,
                                    physical_activity_level, diet_type, smoking_habits,
                                    alcohol_consumption, family_history_obesity, blood_pressure,
                                    cholesterol_levels, education_level, income_level, geographical_region]])

            # Apply the scaler transformation to numerical features (age, gender, height, weight, bmi)
            input_data[:, :4] = scaler.transform(input_data[:, :4])

            # Make prediction using the trained TabNet model
            prediction = transfer_tabnet_model.predict(input_data)

            # Map the predicted integer to obesity status labels
            obesity_labels = ["Underweight", "Normal weight", "Overweight", "Obese"]
            obesity_status = obesity_labels[int(prediction[0])]

            # Generate health suggestions using LangChain
            response = chain.invoke({"obesity_status": 'obesity'})
            health_suggestions = response.get("text", "No suggestions available.")

            # Clean up the response to format it better
            formatted_suggestions = health_suggestions.split('\n\n')

            # Join all suggestions into a single, readable format
            health_suggestions_cleaned = "\n".join([suggestion.strip() for suggestion in formatted_suggestions if suggestion.strip()])


            # Render the result with health suggestions
            return render_template('result.html', obesity_status=obesity_status, health_suggestions=health_suggestions_cleaned)

        except KeyError as e:
            return f"Error: Missing form field {e}", 400

    else:
        return render_template('predict.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/About')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)

