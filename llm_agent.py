# llm_agent.py

from transformers import pipeline
from langchain import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load a local HuggingFace text-generation model (free + no API key)
def load_local_llm():
    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",  # Or use a larger one like flan-t5-large
        max_length=256,
        temperature=0.7
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return llm

# Build prompt and get response
def ask_llm(agent_prompt, user_input, context):
    prompt = PromptTemplate(
        input_variables=["agent_prompt", "context", "question"],
        template="""
        You are a helpful assistant. Use the given context to answer the user's question.

        Agent Instructions: {agent_prompt}
        Context: {context}
        Question: {question}
        Answer:"""
    )
    llm = load_local_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(agent_prompt=agent_prompt, context=context, question=user_input)
    return result
