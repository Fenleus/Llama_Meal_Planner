import gradio as gr
import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
import time
import logging
from typing import Dict, Any, List, Tuple

# LangChain imports
from langchain_community.llms import HuggingFaceTextGenInference
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment setup
LLAMA_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "") or os.environ.get("HF_TOKEN", "")

# NOTE: Remove any hardcoded API tokens from this file to avoid secret leaks.

def get_current_datetime():
    """Get the current date and time in UTC format"""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def get_current_user():
    """Get the current user login"""
    return os.environ.get("USER", "Fenleusyes")

def calculate_bmi_percentile_category(weight_kg: float, height_cm: float, age_months: int) -> str:
    """Calculate BMI and categorize based on pediatric standards"""
    if height_cm <= 0:
        return "Invalid height"
    
    bmi = weight_kg / ((height_cm / 100) ** 2)
    
    if age_months < 24:  # 0-2 years
        if bmi < 14:
            return "Underweight"
        elif bmi < 18:
            return "Normal weight"
        elif bmi < 20:
            return "Overweight"
        else:
            return "Obese"
    else:  # 2-5 years
        if bmi < 13.5:
            return "Underweight"
        elif bmi < 17:
            return "Normal weight"
        elif bmi < 19:
            return "Overweight"
        else:
            return "Obese"

def get_dietary_guidelines(age_months: int, bmi_category: str) -> Dict[str, str]:
    """Get age-appropriate dietary guidelines"""
    guidelines = {
        "0-6": {
            "primary": "Exclusive breastfeeding recommended",
            "notes": "No solid foods, water, or other liquids needed",
            "calories": "500-600 kcal/day from breast milk"
        },
        "6-12": {
            "primary": "Breastfeeding + complementary foods",
            "notes": "Introduce iron-rich foods, variety of textures",
            "calories": "600-900 kcal/day"
        },
        "12-24": {
            "primary": "Family foods + continued breastfeeding",
            "notes": "3 meals + 2 snacks, whole milk until 2 years",
            "calories": "900-1200 kcal/day"
        },
        "24-60": {
            "primary": "Balanced family diet",
            "notes": "Variety of foods, limit processed foods and sugar",
            "calories": "1200-1600 kcal/day"
        }
    }
    
    if age_months < 6:
        return guidelines["0-6"]
    elif age_months < 12:
        return guidelines["6-12"]
    elif age_months < 24:
        return guidelines["12-24"]
    else:
        return guidelines["24-60"]

def create_system_prompt(age_months: int, weight_kg: float, height_cm: float, bmi_category: str) -> str:
    """Create comprehensive system prompt for pediatric dietary suggestions"""
    guidelines = get_dietary_guidelines(age_months, bmi_category)
    age_years = age_months / 12
    
    return f"""You are a specialized pediatric nutrition AI assistant trained on evidence-based dietary guidelines for children aged 0-5 years.
CHILD PROFILE:
- Age: {age_months} months ({age_years:.1f} years)
- Weight: {weight_kg} kg
- Height: {height_cm} cm
- BMI Category: {bmi_category}
DIETARY GUIDELINES FOR THIS AGE GROUP:
- Primary approach: {guidelines['primary']}
- Key considerations: {guidelines['notes']}
- Estimated daily calories: {guidelines['calories']}
SAFETY PROTOCOLS:
- NEVER recommend foods that are choking hazards (nuts, whole grapes, popcorn for under 4 years)
- NO honey for children under 12 months (botulism risk)
- NO whole cow's milk for children under 12 months
- Consider food allergies and introduce new foods gradually
- Prioritize nutrient-dense foods over empty calories
BMI-SPECIFIC ADJUSTMENTS:
- If Underweight: Focus on calorie-dense, nutritious foods; frequent small meals
- If Normal weight: Maintain balanced variety, appropriate portions
- If Overweight/Obese: Emphasize fruits, vegetables, and physical activity; avoid restricting calories severely
RESPONSE FORMAT:
Provide practical, safe meal suggestions with:
1. Age-appropriate foods and textures
2. Portion sizes suitable for the child's age
3. Nutritional benefits
4. Safety considerations
5. Preparation tips for parents
Always recommend consulting with pediatricians for specific concerns.

Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {get_current_datetime()}
Current User's Login: {get_current_user()}
"""

class LlamaModelWrapper:
    """Wrapper for LangChain LLM with Llama-3.2-3B-Instruct"""
    
    def __init__(self):
        """Initialize the wrapper for Llama-3.2-3B-Instruct only"""
        self.model_id = LLAMA_MODEL
    
    def _create_langchain_model(self):
        """Create a LangChain LLM instance for Llama-3.2-3B-Instruct"""
        try:
            # Configure callback manager
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            # Set up the model with LangChain HuggingFace integration
            from langchain_huggingface.chat_models.huggingface import HuggingFaceHub
            
            # Create the model
            llm = HuggingFaceHub(
                repo_id=self.model_id,
                huggingfacehub_api_token=HF_API_TOKEN,
                model_kwargs={
                    "temperature": 0.7,
                    "max_new_tokens": 500,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                }
            )
            
            return llm
        except Exception as e:
            logger.error(f"Error creating Llama model: {str(e)}")
            return None
    
    def generate_response(self, system_prompt: str, user_query: str) -> Tuple[str, str]:
        """Generate response using Llama-3.2-3B-Instruct"""
        try:
            logger.info(f"Using model: {self.model_id}")
            
            # Format prompt using Llama 3.2's specific instruction format
            formatted_prompt = f"""<|system|>
{system_prompt}</s>
<|user|>
{user_query}</s>
<|assistant|>"""
            
            # Create LangChain prompt template
            prompt = PromptTemplate(
                template="{formatted_prompt}",
                input_variables=["formatted_prompt"]
            )
            
            # Create the model
            llm = self._create_langchain_model()
            if not llm:
                logger.error("Failed to create Llama model")
                return None, None
            
            # Create and run the chain
            chain = LLMChain(llm=llm, prompt=prompt)
            response = chain.run(formatted_prompt=formatted_prompt)
            
            # Check if response is meaningful
            if response and len(response.strip()) >= 50:
                return response.strip(), self.model_id
            else:
                logger.warning("Model returned insufficient response")
                return None, None
                
        except Exception as e:
            logger.error(f"Error with Llama model: {str(e)}")
            return None, None

def suggest_meal_online(age_months: int, weight_kg: float, height_cm: float, dietary_request: str) -> str:
    """Generate meal suggestions using Llama-3.2-3B-Instruct"""
    
    if not HF_API_TOKEN:
        return "⚠️ **Setup Required**: Please set your Hugging Face API token as an environment variable named 'HF_API_TOKEN'."
    
    try:
        # Calculate BMI
        bmi = weight_kg / ((height_cm / 100) ** 2)
        bmi_category = calculate_bmi_percentile_category(weight_kg, height_cm, age_months)
        
        # Create system prompt
        system_prompt = create_system_prompt(age_months, weight_kg, height_cm, bmi_category)
        
        # Initialize model wrapper
        model_wrapper = LlamaModelWrapper()
        
        # Generate response
        response_text, model_used = model_wrapper.generate_response(
            system_prompt=system_prompt,
            user_query=f"Please suggest meal options for a {age_months}-month-old child who is {bmi_category}. Specifically: {dietary_request}"
        )
        
        if response_text:
            # Format final response
            final_response = f"""**🧒 Child Profile Summary:**
- **Age:** {age_months} months ({age_months/12:.1f} years)
- **Weight:** {weight_kg} kg
- **Height:** {height_cm} cm  
- **BMI:** {bmi:.1f} ({bmi_category})
**🍽️ Dietary Recommendations:**
{response_text}
**📋 General Guidelines for {age_months}-month-old:**
{get_dietary_guidelines(age_months, bmi_category)['primary']}
- {get_dietary_guidelines(age_months, bmi_category)['notes']}
- {get_dietary_guidelines(age_months, bmi_category)['calories']}
---
*⚠️ **Important:** Always consult with your pediatrician for specific dietary concerns or medical conditions.*
*🤖 **Model Used:** {model_used}*
*⏱️ **Generated at:** {get_current_datetime()} UTC*"""
            
            return final_response
        else:
            # If model failed, return fallback
            return create_fallback_response(age_months, weight_kg, height_cm, bmi_category, dietary_request)
            
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return create_fallback_response(age_months, weight_kg, height_cm, bmi_category, dietary_request)

def create_fallback_response(age_months: int, weight_kg: float, height_cm: float, 
                            bmi_category: str, dietary_request: str) -> str:
    """Create a fallback response when API fails"""
    bmi = weight_kg / ((height_cm / 100) ** 2)
    guidelines = get_dietary_guidelines(age_months, bmi_category)
    
    # Basic meal suggestions based on age
    if age_months < 6:
        suggestions = "Exclusive breastfeeding or formula feeding. No solid foods recommended yet."
    elif age_months < 12:
        suggestions = """
        - Iron-fortified cereals mixed with breast milk/formula
        - Pureed fruits: banana, apple, pear
        - Pureed vegetables: sweet potato, carrots, green beans
        - Soft finger foods: well-cooked pasta, soft fruits
        - Avoid honey, whole nuts, choking hazards
        """
    elif age_months < 24:
        suggestions = """
        - Soft table foods cut into small pieces
        - Whole milk (after 12 months)
        - Soft fruits and vegetables
        - Well-cooked grains and pasta
        - Soft-cooked eggs, fish, poultry
        - Avoid hard candies, whole grapes, nuts
        """
    else:
        suggestions = """
        - Family meals with appropriate portions
        - Variety of fruits and vegetables
        - Whole grains and lean proteins
        - Dairy products for calcium
        - Limit processed foods and added sugars
        - Encourage self-feeding and food exploration
        """
    
    return f"""**🧒 Child Profile Summary:**
- **Age:** {age_months} months ({age_months/12:.1f} years)
- **Weight:** {weight_kg} kg
- **Height:** {height_cm} cm  
- **BMI:** {bmi:.1f} ({bmi_category})
**🍽️ Age-Appropriate Meal Suggestions:**
{suggestions}
**📋 Guidelines for This Age Group:**
- **Primary approach:** {guidelines['primary']}
- **Key considerations:** {guidelines['notes']}
- **Estimated daily calories:** {guidelines['calories']}
**🔧 Note:** Llama model temporarily unavailable - showing evidence-based guidelines instead.
---
*⚠️ **Important:** Always consult with your pediatrician for specific dietary concerns or medical conditions.*
*⏱️ **Generated at:** {get_current_datetime()} UTC*"""

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="🦙 Llama Meal Planner", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🦙 Llama Meal Planner")
        gr.Markdown("**AI-powered dietary suggestions for children 0-5 years**")
        gr.Markdown(f"Powered by {LLAMA_MODEL}")
        
        # Display current date/time
        datetime_display = gr.Markdown(f"**Current Date/Time (UTC):** {get_current_datetime()}")
        
        # Update the datetime every 60 seconds
        def update_datetime():
            return f"**Current Date/Time (UTC):** {get_current_datetime()}"
        
        # Removed datetime_display.every(60, update_datetime) because gr.Markdown has no 'every' method
        
        with gr.Row():
            with gr.Column(scale=1):
                age_input = gr.Slider(
                    minimum=0, 
                    maximum=60, 
                    value=24, 
                    step=1, 
                    label="👶 Child's Age (months)",
                    info="0-60 months (0-5 years)"
                )
                weight_input = gr.Number(
                    value=12.0, 
                    label="⚖️ Weight (kg)", 
                    minimum=2.0, 
                    maximum=30.0,
                    info="Child's current weight"
                )
                height_input = gr.Number(
                    value=85.0, 
                    label="📏 Height (cm)", 
                    minimum=45.0, 
                    maximum=120.0,
                    info="Child's current height"
                )
                request_input = gr.Textbox(
                    label="🍽️ What do you need help with?", 
                    placeholder="e.g., 'Suggest a healthy breakfast for my toddler' or 'What snacks help with weight gain?'",
                    lines=3,
                    info="Describe what kind of meal suggestions you need"
                )
                
                submit_btn = gr.Button("🔍 Get Meal Suggestions", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                output = gr.Textbox(
                    label="📋 Meal Suggestions", 
                    lines=15,
                    max_lines=25,
                    info="AI-generated dietary recommendations"
                )
        
        # Example requests
        gr.Markdown("### 💡 Try These Examples:")
        examples = [
            [18, 10.5, 80, "Suggest a healthy breakfast for my 18-month-old"],
            [36, 15, 95, "What snacks are good for my 3-year-old's growth?"],
            [8, 8.5, 70, "Safe finger foods for my 8-month-old baby"],
            [48, 18, 105, "Weekly meal plan for my 4-year-old"],
            [30, 11, 85, "Healthy foods to help my underweight toddler gain weight"]
        ]
        
        gr.Examples(
            examples=examples,
            inputs=[age_input, weight_input, height_input, request_input],
            outputs=output,
            fn=suggest_meal_online,
            cache_examples=False
        )
        
        submit_btn.click(
            suggest_meal_online,
            inputs=[age_input, weight_input, height_input, request_input],
            outputs=output
        )
        
        # Disclaimer
        gr.Markdown("""
        ### ⚠️ Important Medical Disclaimer
        This tool provides general dietary suggestions based on established pediatric nutrition guidelines. 
        **Always consult with your pediatrician, registered dietitian, or healthcare provider for:**
        - Specific medical conditions or allergies
        - Growth concerns or developmental delays
        - Personalized nutrition plans
        - Any serious dietary restrictions
        
        ### 🔧 About This App
        - **Powered by:** Llama 3.2 3B Instruct model exclusively
        - **Guidelines:** Based on WHO, CDC, and AAP recommendations
        - **Age Range:** 0-5 years (0-60 months)
        - **Features:** BMI calculation, safety protocols, age-appropriate suggestions
        """)
    
    return interface

# Launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch()