# 🦙 Llama Meal Planner

An AI-powered application that provides evidence-based dietary suggestions for children aged 0-5 years using Meta's Llama-3.2-3B-Instruct model.

## ✨ Features

- 🧠 **Powered by Llama-3.2-3B-Instruct**: Uses Meta's latest instruction-following model for accurate responses
- 📊 **BMI calculation and categorization** based on pediatric standards
- 🍼 **Age-appropriate dietary guidelines** (0-60 months)
- 🤖 **AI-generated meal suggestions** with intelligent fallback system
- 🛡️ **Safety protocols** for different age groups (choking hazards, allergies)
- ✅ **Graceful error handling** and input validation
- 🎨 **Interactive Gradio interface** with examples and real-time suggestions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Hugging Face API token (for Llama model access)

### Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/Fenleus/Llama_Meal_Planner.git
   cd Llama_Meal_Planner
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Hugging Face token**
   
   Create a `.env` file in the project root:
   ```bash
   HF_API_TOKEN=your_hugging_face_token_here
   ```
   
   **Important**: You need to:
   - Create a free account at [Hugging Face](https://huggingface.co)
   - Generate an API token in your [HF Settings](https://huggingface.co/settings/tokens)
   - Accept the license for [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

4. **Run the application**
   ```bash
   python llama_bmi_meal.py
   ```

5. **Open your browser** and navigate to `http://localhost:7860`

## 🎯 Usage

### Input Parameters
- **👶 Child's Age**: 0-60 months (0-5 years)
- **⚖️ Weight**: Child's current weight in kg
- **📏 Height**: Child's current height in cm  
- **🍽️ Request**: What kind of meal suggestions you need

### Example Queries
- "Suggest a healthy breakfast for my 18-month-old"
- "What snacks are good for my 3-year-old's growth?"
- "Safe finger foods for my 8-month-old baby"
- "Weekly meal plan for my 4-year-old"
- "Healthy foods to help my underweight toddler gain weight"

### Output
The app provides:
- **Child profile summary** with BMI calculation
- **AI-generated dietary recommendations** from Llama-3.2-3B-Instruct
- **Age-appropriate guidelines** based on WHO/CDC/AAP standards
- **Safety considerations** and preparation tips
- **Fallback recommendations** if the AI model is unavailable

## 🔧 Technical Details

### Model Integration
- **Model**: `meta-llama/Llama-3.2-3B-Instruct`
- **Framework**: LangChain with HuggingFace integration
- **Prompt Format**: Llama instruction-following format with system/user/assistant structure
- **Parameters**: 
  - Temperature: 0.7
  - Max tokens: 500
  - Top-p: 0.9
  - Repetition penalty: 1.1

### Architecture
```
User Input → BMI Calculation → System Prompt Generation → 
Llama Model → Response Processing → Formatted Output
                ↓ (if model fails)
         Evidence-based Fallback Response
```

### Dependencies
- `gradio>=3.50.0` - Web interface
- `langchain>=0.0.200` - LLM orchestration
- `langchain-huggingface>=0.0.10` - HF integration
- `langchain-community>=0.0.20` - Community tools
- `requests>=2.31.0` - HTTP requests
- `python-dotenv>=1.0.0` - Environment variables

## 🛡️ Safety & Guidelines

### Built-in Safety Features
- **Age-appropriate food recommendations** 
- **Choking hazard prevention** (no nuts, whole grapes for <4 years)
- **Allergy considerations** (no honey for <12 months)
- **Portion size guidance** based on developmental stage
- **Nutritional balance** prioritization

### Medical Disclaimer
⚠️ **Important**: This tool provides general dietary suggestions based on established pediatric nutrition guidelines. Always consult with your pediatrician, registered dietitian, or healthcare provider for:
- Specific medical conditions or allergies
- Growth concerns or developmental delays  
- Personalized nutrition plans
- Any serious dietary restrictions

## 🧪 Testing

Run the comprehensive test suite:
```bash
python -m pytest tests/ -v
```

Or run manual tests:
```bash
python /tmp/comprehensive_test.py
```

## 📱 Screenshots

![Llama Meal Planner Interface](https://github.com/user-attachments/assets/e61fb8d2-4ef7-43b1-893b-bf7a56792054)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Meta AI** for the Llama-3.2-3B-Instruct model
- **Hugging Face** for model hosting and API
- **WHO, CDC, AAP** for pediatric nutrition guidelines
- **Gradio** for the user-friendly interface framework