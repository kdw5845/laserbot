import streamlit as st

# Initialize session state to store conversation history
if 'history' not in st.session_state:
    st.session_state.history = []
    st.session_state.input_key = 0

    # We also need to build the variables needed for the chat and load the vector store
    # Import needed packages
    #
    # Standard Stuff
    #
    import os
    import sys

    # We use itemgetter in our chains
    #
    from   operator  import itemgetter

    # A lot of stuff from langchain 
    #
    from   langchain.prompts import PromptTemplate,MessagesPlaceholder
    from   langchain.prompts.chat import ChatPromptTemplate
    from   langchain.chains import LLMChain
    from   langchain_community.vectorstores import FAISS
    from   langchain.chains import RetrievalQA
    from   langchain_community.docstore.document import Document
    from   langchain.chains import ConversationalRetrievalChain
    from   langchain.memory import ConversationBufferMemory
    from   langchain.schema import format_document
    from   langchain.schema.messages import AIMessage, HumanMessage, get_buffer_string
    from   langchain_core.runnables import RunnableParallel
    from   langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from   langchain_openai import ChatOpenAI
    from   openai import OpenAI
    from   langchain_openai import OpenAIEmbeddings

    # NumPy
    #
    import numpy as np

    os.environ['OPENAI_API_KEY'] = st.secrets.ai_credentials.openai_api_key
    st.session_state.model_kwargs = { 'top_p' : 0.1 }
    st.session_state.llm = ChatOpenAI(temperature=0.1, model_name="gpt-4o", model_kwargs = st.session_state.model_kwargs)
 #  st.session_state.llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo", model_kwargs = st.session_state.model_kwargs)
    EMBEDDING_MODEL_ID = 'text-embedding-3-large'
    st.session_state.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_ID)

    # Function to load vector store
    def load_local_vector_store(vector_store_path):
        with st.spinner(text="Initializing"):
            try:
                with open(f"{vector_store_path}/embeddings_model_id", 'r') as f:
                    embeddings_model_id = f.read()
                vector_store = FAISS.load_local(vector_store_path, st.session_state.embeddings)
                return vector_store
            except Exception:
                st.stop()

    # Load vector storage from disk
    #
    st.session_state.vector_store_path='./open_ai_data'
    st.session_state.db = load_local_vector_store(st.session_state.vector_store_path)
    st.session_state.retriever = st.session_state.db.as_retriever(search_type="similarity", search_kwargs={"k": 8})

    st.session_state.qa_template = """
    You are a professional laser technician with years of experience in laser hair removal and laser tattoo removal as well
    as using IPL for facial skin rejuvination. You are assisting a student in laser hair removal to learn laser techniques
    and help her passing the test for laser technician. The student doesn't speak english as a first language and has very
    little knowledge about physics. Hence, you have to use words and explanations that match your students level of understanding.
    For instance, you don't just say 'nm' or 'naometer' without giving addition explanations to the student. It is very
    important that you take a step back and rethink your answers or advise. Be an excellent teacher and educator. Having
    said this; given the following [Chat History] and provided [Context], please respond to the student's [Question]. Don't 
    preceede your answer with '[Answer] :' or similar prefixes. Also, don't use '###' or '**' when you give a list of answers
    but use numbers or indentations instead. make your output appealing!

    [Chat History]: {chat_history}

    [Context]: {context}

    [Question]: {question}
    """

    st.session_state.QA_PROMPT       = ChatPromptTemplate.from_template(st.session_state.qa_template)
    st.session_state.DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

    def _combine_documents(docs, document_prompt=st.session_state.DOCUMENT_PROMPT, document_separator="\n\n"):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    st.session_state.memory = ConversationBufferMemory(llm=st.session_state.llm, return_messages=True, output_key="answer", input_key="question")

    st.session_state.loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(st.session_state.memory.load_memory_variables) | itemgetter("history")
    )

    st.session_state.retrieved_documents = {
        "docs"        : itemgetter("question") | st.session_state.retriever,   # Please note that all second entries need to be callables(!)
        "question"    : lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"]
    }

    st.session_state.final_inputs = {
        "context"     : lambda x: _combine_documents(x["docs"]),
        "question"    : itemgetter("question"),
        "chat_history": itemgetter("chat_history")
    }

    st.session_state.answer = {
        "answer"  : st.session_state.final_inputs | st.session_state.QA_PROMPT | st.session_state.llm,
        "question": itemgetter("question"),
    }

    st.session_state.final_chain = st.session_state.loaded_memory | st.session_state.retrieved_documents | st.session_state.answer


st.session_state.col1 = st.columns(1)

# Main display area for text
st.text_area("Conversation - please scroll to see latest answer", value="\n".join(st.session_state.history), height=550, key='conversation_{st.session_state.input_key}', disabled=False)

# Function to update conversation history
def update_history():
    #
    # Prepare the input for the RAG model
    user_input  = st.session_state[f'input_{st.session_state.input_key}']
    inputs = {"question": user_input}

    # Invoke the RAG model to get an answer
    result = st.session_state.final_chain.invoke(inputs)
    
    # Save the current question and its answer to memory for future context
    #
    st.session_state.memory.save_context(inputs, {"answer": result["answer"].content})

    # Update conversation history
    st.session_state.history.extend([f"You: {user_input}", " ", result["answer"].content, " "])
    st.session_state.input_key += 1

# User input
st.text_input("Please state your question here - Press ENTER when you are finished", key=f'input_{st.session_state.input_key}', on_change=update_history, max_chars=200)  # Adjust max_chars as needed

st.write("")
st.write("")
st.image('avatar.png', caption='Laser Hair Removal Expert')
st.write("For educational purposes only!")
st.write("Chatbot may give wrong answers!")

