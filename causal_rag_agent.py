"""
Causal RAG Agent: A Retrieval-Augmented Generation system for causal reasoning and prescriptive guidance.

This module implements a complete Causal RAG Agent that:
1. Stores causal knowledge in a ChromaDB Vector Store
2. Uses LangChain to orchestrate retrieval and generation
3. Provides prescriptive guidance based on causal reasoning
4. Offers a Streamlit UI for user interaction
"""

import os
import chromadb
from chromadb.config import Settings
import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import streamlit as st


# Causal knowledge base - simulated causal statements
CAUSAL_STATEMENTS = [
    {
        "statement": "Policy 101 states that a price decrease of 10% or more leads to a temporary sales spike of 20-30% within the first week, followed by normalization.",
        "domain": "pricing",
        "cause": "price decrease",
        "effect": "temporary sales spike"
    },
    {
        "statement": "System failure in Region X is caused by vendor Y outage. When vendor Y experiences downtime, Region X loses connectivity within 5 minutes.",
        "domain": "infrastructure",
        "cause": "vendor Y outage",
        "effect": "Region X system failure"
    },
    {
        "statement": "Customer churn increases by 15% when support response time exceeds 24 hours. Fast support response (under 6 hours) reduces churn by 40%.",
        "domain": "customer service",
        "cause": "slow support response",
        "effect": "increased customer churn"
    },
    {
        "statement": "Marketing campaign effectiveness drops by 50% when ad frequency exceeds 5 impressions per user per day, causing ad fatigue.",
        "domain": "marketing",
        "cause": "high ad frequency",
        "effect": "ad fatigue and reduced effectiveness"
    },
    {
        "statement": "Product quality issues in manufacturing line B are caused by temperature fluctuations above 2 degrees Celsius. Maintaining stable temperature reduces defects by 80%.",
        "domain": "manufacturing",
        "cause": "temperature fluctuations",
        "effect": "product quality issues"
    },
    {
        "statement": "Website conversion rate increases by 25% when page load time is reduced below 2 seconds. Every additional second of load time decreases conversions by 7%.",
        "domain": "web performance",
        "cause": "slow page load time",
        "effect": "reduced conversion rate"
    },
    {
        "statement": "Employee productivity decreases by 30% when meeting time exceeds 3 hours per day. Optimal meeting time is 1-2 hours daily for maximum productivity.",
        "domain": "workplace",
        "cause": "excessive meetings",
        "effect": "reduced employee productivity"
    },
    {
        "statement": "Inventory shortage in warehouse C occurs when supplier D delays shipments by more than 3 days. Implementing buffer stock of 20% prevents 90% of shortages.",
        "domain": "supply chain",
        "cause": "supplier delays",
        "effect": "inventory shortage"
    },
    {
        "statement": "Software deployment failures increase by 60% when code review coverage falls below 80%. Comprehensive code reviews with 95%+ coverage reduce failures by 75%.",
        "domain": "software engineering",
        "cause": "insufficient code review",
        "effect": "deployment failures"
    },
    {
        "statement": "Sales team performance drops by 40% when training frequency is less than once per quarter. Monthly training sessions improve performance by 35% and close rates by 20%.",
        "domain": "sales",
        "cause": "infrequent training",
        "effect": "reduced sales performance"
    }
]


# Custom Causal Prompt Template
CAUSAL_PROMPT_TEMPLATE = """You are an expert causal reasoning analyst. Your role is to analyze problems using retrieved causal knowledge and provide prescriptive guidance.

Context (Retrieved Causal Facts):
{context}

User Question: {question}

Instructions:
1. IDENTIFY: Carefully examine the retrieved causal facts above. List the relevant causal relationships that apply to the user's question.

2. REASON: Based on the identified causal facts, explain the likely cause(s) of the problem mentioned in the user's question. Use the causal links to build a logical explanation.

3. PRESCRIBE: Provide concrete, actionable steps that address the root cause. Your recommendations should be specific and directly derived from the causal relationships identified.

Format your answer as follows:
📊 IDENTIFIED CAUSAL FACTS:
[List the relevant causal relationships from the context]

🔍 CAUSAL REASONING:
[Explain why the problem is occurring based on the causal facts]

💡 PRESCRIPTIVE ACTIONS:
[Provide specific, actionable recommendations]

Answer:"""


def initialize_vector_store():
    """
    Initialize ChromaDB Vector Store and populate it with causal statements.
    
    Returns:
        Chroma: Configured ChromaDB vector store with causal knowledge
    """
    # Create documents from causal statements
    documents = []
    for item in CAUSAL_STATEMENTS:
        doc_text = f"{item['statement']} (Domain: {item['domain']}, Cause: {item['cause']}, Effect: {item['effect']})"
        doc = Document(
            page_content=doc_text,
            metadata={
                "domain": item["domain"],
                "cause": item["cause"],
                "effect": item["effect"]
            }
        )
        documents.append(doc)
    
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create ChromaDB vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="causal_knowledge",
        persist_directory="./chroma_db"
    )
    
    return vectorstore


def create_causal_rag_chain(vectorstore):
    """
    Create a LangChain RetrievalQA chain with custom causal prompt template.
    
    Args:
        vectorstore: ChromaDB vector store containing causal knowledge
        
    Returns:
        RetrievalQA: Configured RAG chain for causal reasoning
    """
    # Initialize LLM (GPT-3.5-turbo)
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3,  # Lower temperature for more focused reasoning
        max_tokens=1000
    )
    
    # Create custom prompt template
    prompt = PromptTemplate(
        template=CAUSAL_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    # Create retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant causal facts
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain


def main():
    """
    Main Streamlit application for Causal RAG Agent.
    """
    st.set_page_config(
        page_title="Causal RAG Agent",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Causal RAG Agent")
    st.markdown("### AI-Powered Causal Reasoning & Prescriptive Guidance")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to use the Causal RAG Agent"
        )
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ Please enter your OpenAI API Key")
        
        st.markdown("---")
        st.header("📚 Knowledge Base")
        st.markdown(f"**Causal Statements:** {len(CAUSAL_STATEMENTS)}")
        st.markdown("**Domains Covered:**")
        domains = list(set([item["domain"] for item in CAUSAL_STATEMENTS]))
        for domain in sorted(domains):
            st.markdown(f"- {domain.title()}")
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        This Causal RAG Agent combines:
        - **Vector Database** (ChromaDB) for storing causal knowledge
        - **LangChain** for orchestration
        - **GPT-3.5** for reasoning and generation
        
        It provides prescriptive guidance by:
        1. Retrieving relevant causal facts
        2. Reasoning about the problem
        3. Prescribing actionable solutions
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🤔 Ask a Question")
        
        # Example questions
        with st.expander("💡 Example Questions"):
            st.markdown("""
            - Why did our sales drop last month?
            - What's causing the system failures in Region X?
            - How can we reduce customer churn?
            - Why is our marketing campaign not performing well?
            - What's causing quality issues in our manufacturing?
            - How can we improve website conversions?
            - Why is employee productivity declining?
            - What's causing inventory shortages?
            - How can we reduce deployment failures?
            - Why is our sales team underperforming?
            """)
        
        # User input
        user_question = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., Why did our sales drop last month?"
        )
        
        submit_button = st.button("🚀 Get Causal Analysis", type="primary", use_container_width=True)
    
    with col2:
        st.header("📊 Process")
        st.markdown("""
        **Step 1:** 🔎 Retrieve  
        Find relevant causal facts
        
        **Step 2:** 🧠 Reason  
        Analyze the causal links
        
        **Step 3:** 💊 Prescribe  
        Provide actionable steps
        """)
    
    # Process user question
    if submit_button and user_question:
        if not api_key:
            st.error("❌ Please enter your OpenAI API Key in the sidebar")
        else:
            with st.spinner("🔄 Analyzing causal relationships..."):
                try:
                    # Initialize vector store
                    vectorstore = initialize_vector_store()
                    
                    # Create RAG chain
                    qa_chain = create_causal_rag_chain(vectorstore)
                    
                    # Get answer
                    result = qa_chain({"query": user_question})
                    
                    # Display results
                    st.markdown("---")
                    st.header("📋 Causal Analysis Results")
                    
                    # Main answer
                    st.markdown(result["result"])
                    
                    # Source documents
                    with st.expander("📚 View Retrieved Causal Facts (Source Documents)"):
                        for i, doc in enumerate(result["source_documents"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.info(doc.page_content)
                            st.markdown(f"*Domain: {doc.metadata['domain']}*")
                            st.markdown("---")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.markdown("""
                    **Troubleshooting:**
                    - Verify your OpenAI API Key is correct
                    - Check your internet connection
                    - Ensure you have sufficient API credits
                    """)
    
    elif submit_button and not user_question:
        st.warning("⚠️ Please enter a question before submitting")


if __name__ == "__main__":
    main()
