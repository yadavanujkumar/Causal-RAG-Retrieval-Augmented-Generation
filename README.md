# Causal RAG Agent: Retrieval-Augmented Generation for Causal Reasoning

🔍 **An AI-powered system that answers "why" and "what should be done" questions using causal knowledge and prescriptive reasoning.**

## Overview

The Causal RAG Agent is a sophisticated AI system that combines:
- **Vector Database** (ChromaDB) for storing causal knowledge
- **LangChain** for orchestrating retrieval and generation
- **Large Language Model** (GPT-3.5) for causal reasoning
- **Streamlit UI** for interactive user experience

### Key Features

✅ **Causal Knowledge Base**: Pre-populated with 10 causal statements across multiple domains  
✅ **Three-Step Reasoning**: Identify → Reason → Prescribe  
✅ **Prescriptive Guidance**: Actionable recommendations based on causal analysis  
✅ **Interactive UI**: User-friendly Streamlit interface  
✅ **Source Transparency**: View retrieved causal facts supporting each answer  

## Architecture

```
User Question
     ↓
[Streamlit UI]
     ↓
[LangChain RAG Chain]
     ↓
[ChromaDB Vector Store] → [Retrieve Top-K Causal Facts]
     ↓
[Custom Causal Prompt Template]
     ↓
[GPT-3.5 Turbo] → [Generate Causal Analysis]
     ↓
[Prescriptive Answer]
```

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API Key ([Get one here](https://platform.openai.com/api-keys))

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/yadavanujkumar/Causal-RAG-Retrieval-Augmented-Generation.git
cd Causal-RAG-Retrieval-Augmented-Generation
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
streamlit run causal_rag_agent.py
```

This will start the Streamlit server and open the application in your default web browser (typically at `http://localhost:8501`).

### Using the Application

1. **Enter your OpenAI API Key** in the sidebar
2. **Type your question** in the main input area
3. **Click "Get Causal Analysis"** to receive your answer
4. **Review the results**:
   - Identified causal facts
   - Causal reasoning explanation
   - Prescriptive actions
   - Source documents used

### Example Questions

Try asking questions like:
- "Why did our sales drop last month?"
- "What's causing the system failures in Region X?"
- "How can we reduce customer churn?"
- "Why is our marketing campaign not performing well?"
- "What's causing quality issues in our manufacturing?"
- "How can we improve website conversions?"
- "Why is employee productivity declining?"

## Technical Details

### Causal Knowledge Base

The system includes 10 pre-configured causal statements covering:
- **Pricing**: Price changes and sales impact
- **Infrastructure**: System failures and vendor dependencies
- **Customer Service**: Support response time and churn
- **Marketing**: Ad frequency and campaign effectiveness
- **Manufacturing**: Temperature control and quality
- **Web Performance**: Load time and conversion rates
- **Workplace**: Meeting time and productivity
- **Supply Chain**: Supplier delays and inventory
- **Software Engineering**: Code review and deployment success
- **Sales**: Training frequency and performance

### Custom Prompt Template

The system uses a specialized prompt template that instructs the LLM to:

1. **IDENTIFY**: List relevant causal facts from retrieved context
2. **REASON**: Explain the problem using causal relationships
3. **PRESCRIBE**: Provide actionable recommendations

This three-step approach ensures comprehensive causal analysis and practical guidance.

### Components

- **`initialize_vector_store()`**: Creates and populates ChromaDB with causal knowledge
- **`create_causal_rag_chain()`**: Configures LangChain RetrievalQA with custom prompt
- **`main()`**: Streamlit UI and application logic

### Configuration

- **LLM Model**: GPT-3.5-turbo
- **Temperature**: 0.3 (focused reasoning)
- **Max Tokens**: 1000
- **Retrieval**: Top 3 most similar causal facts
- **Embedding Model**: OpenAI text-embedding-ada-002

## Project Structure

```
Causal-RAG-Retrieval-Augmented-Generation/
├── causal_rag_agent.py      # Main application script
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── README.md                # This file
└── chroma_db/              # Vector database (created at runtime)
```

## Dependencies

- **langchain**: LLM orchestration framework
- **langchain-openai**: OpenAI integration for LangChain
- **chromadb**: Vector database for causal knowledge
- **pandas**: Data manipulation
- **streamlit**: Web UI framework
- **openai**: OpenAI API client
- **tiktoken**: Token counting for OpenAI models

## Extending the System

### Adding New Causal Statements

Edit the `CAUSAL_STATEMENTS` list in `causal_rag_agent.py`:

```python
CAUSAL_STATEMENTS.append({
    "statement": "Your causal statement here",
    "domain": "your_domain",
    "cause": "the cause",
    "effect": "the effect"
})
```

### Customizing the Prompt

Modify `CAUSAL_PROMPT_TEMPLATE` to adjust the reasoning structure or output format.

### Changing the LLM

Replace `ChatOpenAI` with other LangChain-compatible models:

```python
from langchain.chat_models import ChatAnthropic  # Example
llm = ChatAnthropic(model="claude-3-opus-20240229")
```

## Troubleshooting

### Common Issues

**Issue**: "OpenAI API Key not configured"  
**Solution**: Ensure you've entered a valid API key in the sidebar

**Issue**: "Rate limit exceeded"  
**Solution**: Check your OpenAI API usage limits and billing

**Issue**: "Module not found"  
**Solution**: Reinstall dependencies: `pip install -r requirements.txt`

**Issue**: ChromaDB persistence errors  
**Solution**: Delete the `chroma_db` folder and restart the application

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue on GitHub.

## Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Powered by [OpenAI GPT-3.5](https://openai.com/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)
- UI framework by [Streamlit](https://streamlit.io/)