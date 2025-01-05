# Imports

from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
import textwrap
import gradio as gr
import langid
from iso639 import Lang
import os
import torch

# Set up Text Splitter

def setup_text_splitter(split_separator, split_chunk_size, split_chunk_overlap_size, split_length_function):

    text_splitter = CharacterTextSplitter(
        separator = split_separator,
        chunk_size = split_chunk_size,
        chunk_overlap = split_chunk_overlap_size,
        length_function = split_length_function)

    return text_splitter

# Load the external database for RAG and setting up Embedding

def load_and_process_data(dataset_name, page_content_column, text_splitter):

    loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
    data = loader.load()
    split_data = text_splitter.split_documents(data)

    return split_data

def setup_embedding(embedding_model_choice, embed_device_choice, embed_normalization_option):

    hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_choice,
        model_kwargs = {'device': embed_device_choice},
        encode_kwargs = {'normalize_embeddings': embed_normalization_option}
                                         )

    return hf_embeddings

def setup_vectordb_retriever(split_data, hf_embeddings, persist_directory_location, retrieve_k_choice, retrieve_search_type_choice):

    vectordb = Chroma.from_documents(
    documents=split_data,
    embedding=hf_embeddings,
    persist_directory=persist_directory_location
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": retrieve_k_choice}, search_type=retrieve_search_type_choice)

    return retriever

def setup_memory(memory_key_name, memory_input_name, memory_output_name, memory_return_message_option):

    memory = ConversationBufferMemory(
    memory_key = memory_key_name,
    input_key = memory_input_name,
    output_key = memory_output_name,
    return_messages = memory_return_message_option
    )

    return memory

def setup_ollama_model(ollama_model_choice, ollama_temp):

    llm_chosen = OllamaLLM(model = ollama_model_choice, 
                           temperature = ollama_temp,
                           base_url="http://ollama:11434"  # Explicitly point to the Ollama container
    )
    return llm_chosen

def setup_prompt(base_prompt_template, prompt_input_list):

    base_prompt = PromptTemplate(
            template = base_prompt_template,
            input_variables = prompt_input_list)

    return base_prompt

def build_rag_chain(llm_chosen, retriever, memory, chain_return_source_option, chain_return_generate_quest_option, chain_verbose_option, base_prompt):

    llm_with_rag_chain_and_memory = ConversationalRetrievalChain.from_llm(
        llm = llm_chosen,
        retriever = retriever,
        memory = memory,
        return_source_documents = chain_return_source_option,
        return_generated_question = chain_return_generate_quest_option,
        verbose = chain_verbose_option,
        combine_docs_chain_kwargs = {'prompt': base_prompt}
        )

    return llm_with_rag_chain_and_memory

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def format_response_with_source_and_memory(llm_response):
    # Initialize an empty list to collect all parts of the output
    output = []

    # Add the answer
    output.append('\n\n==================== Chatbot Answer:====================')
    output.append(wrap_text_preserve_newlines(llm_response['answer']))

    # Add sources
    output.append('\n\n====================Other Relevant Information and Sources:====================')
    for source in llm_response["source_documents"]:
        output.append(source.metadata['question'])
        output.append(source.page_content)

    # Add history
    output.append('\n\n====================Chat History:====================')
    for history in llm_response['chat_history']:
        output.append(history.content)

    # Combine all parts into a single string and return
    return '\n'.join(output)

def detect_language(input_string):

    #detect the language
    input_lang_code = langid.classify(input_string)[0]

    #convert ISO 639 lang code to major language
    input_language = Lang(input_lang_code).name

    return input_language

def talk_to_chatbot(input_question):

    input_language = detect_language(input_question)

    if input_language != "English":
        # print(f"Translating from {input_language} to English...")
        input_question = llm_chosen.invoke(f"translate this {input_language} content to English: {input_question}")

    # print("Retrieving Information...")
    llm_response = llm_with_rag_chain_and_memory.invoke(input_question)
    chatbot_answer = format_response_with_source_and_memory(llm_response)

    if input_language != "English":
        # print(f"translating from English to {input_language}...")
        chatbot_answer = llm_chosen.invoke(f"translate this English content to {input_language}: {chatbot_answer}")

    return chatbot_answer

def clear_chat_history(clear_memory=True):
    if clear_memory:
        return memory.clear()

# Set Variables

dataset_name = "MakTek/Customer_support_faqs_dataset"
page_content_column = "answer"
split_separator = "\n"
split_chunk_size = 1000
split_chunk_overlap_size = 150
split_length_function = len

embedding_model_choice = "hkunlp/instructor-large"
embed_device_choice = os.getenv("EMBED_DEVICE_CHOICE", "cpu")
# Fallback to CPU if MPS is unavailable
if embed_device_choice == "mps" and not torch.backends.mps.is_available():
    print("MPS availability: ", torch.backends.mps.is_available())
    print("MPS device not available. Falling back to CPU.")
    print("Torch version: ", torch.__version__)
    embed_device_choice = "cpu"
if embed_device_choice == "cuda" and not torch.cuda.is_available():
    print("CUDA device not available. Falling back to CPU.")
    embed_device_choice = "cpu"
print(f"Using device: {embed_device_choice}")
# embed_device_choice = "cpu"
# embed_device_choice = "cuda"
# embed_device_choice = "mps"

embed_normalization_option = True

persist_directory_location = 'docs/chroma/'
retrieve_k_choice = 3
retrieve_search_type_choice = "mmr"

memory_key_name = "chat_history"
memory_input_name = "question"
memory_output_name = "answer"
memory_return_message_option = True

#ollama_model_choice = "llama3.2"
ollama_model_choice = "wangshenzhi/llama3-8b-chinese-chat-ollama-q4"
ollama_temp = 0.1

base_prompt_template = """System: You are a ABC-Company customer service representative.
\n\nInstruction: Answer the customer's question based on following context and chat history if you know the answer. Otherwise, end the answer with 'I am not sure about the answer, please contact our human service for assistance. Thank You!'.
\n\nContext: {context}
\n\nChat history: {chat_history}
\n\nQuestion: {question}
\n\nOutput Answer: """
prompt_input_list = ["context", "question", "chat_history"]

chain_return_source_option = True
chain_return_generate_quest_option = True
chain_verbose_option = False

text_splitter = setup_text_splitter(split_separator, split_chunk_size, split_chunk_overlap_size, split_length_function)
split_data = load_and_process_data(dataset_name, page_content_column, text_splitter)
hf_embeddings = setup_embedding(embedding_model_choice, embed_device_choice, embed_normalization_option)
retriever = setup_vectordb_retriever(split_data, hf_embeddings, persist_directory_location, retrieve_k_choice, retrieve_search_type_choice)
memory = setup_memory(memory_key_name, memory_input_name, memory_output_name, memory_return_message_option)
llm_chosen = setup_ollama_model(ollama_model_choice, ollama_temp)
base_prompt = setup_prompt(base_prompt_template, prompt_input_list)
llm_with_rag_chain_and_memory = build_rag_chain(llm_chosen, retriever, memory, chain_return_source_option, chain_return_generate_quest_option, chain_verbose_option, base_prompt)

memory.clear()

#memory.chat_memory.messages

"""# Gradio Application Build"""

set_gradio_theme = gr.themes.Glass(primary_hue="orange", secondary_hue="gray").set(
    button_primary_background_fill="orange",
    button_primary_background_fill_hover="green",
)

with gr.Blocks(theme=set_gradio_theme) as demo:

    gr.Markdown(
    """
    # Welcome visitor to the our Multilingual Customer Service Chatbot!
    ## I am a demo. Feel free to ask me any questions related to your order and our company in your own language.
    ### I can speak most major languages such as English, Chinese, French, Spanish, Japanese etc...

    ### I am built using Ollama-llama3 llm model fine-tuned by Wangshenzhi and Langchain for RAG (Retrieval-Augmented Generation).
    ### For technical details, please see info at the bottom of the page.

    Start talking to me by typing below.

    Please note that:
    - The output sources and chat-history are mostly for debug and monitor purposes during development. It is for making sure the chatbot is responding properly.
    - The application is running on GPU, so the response time is pretty fast, but multilingual processing can take slightly longer than English.
    """)

    question = gr.Textbox(label="Ask me a question (You can ask in your own language!)", placeholder="Can I request a refund?")
    send_btn = gr.Button("Send Question")
    answer = gr.Textbox(label="Chatbot response", lines=20)

    send_btn.click(fn=talk_to_chatbot, inputs=question, outputs=answer, api_name="customer_service_chatbot")

    gr.Markdown(
    """
    If clear chat history, the next query's chat history will be emptined and refreshed.
    """)
    clear_btn = gr.Button("Clear Chat History")
    clear_btn.click(fn=clear_chat_history)


    gr.Markdown(
    """
    ## Chatbot Technical Details:

    #### Model: llama3-8b-chinese-chat-ollama-q4(8B parameters)
    #### Dataset: Hugging Face Hub "MakTek/Customer_support_faqs_dataset"
    #### Embedding: Hugging Face Hub "hkunlp/instructor-large"
    #### Vector Database: Chroma
    #### Retrieval Search Type: Maximal Marginal Relevance (MMR)
    #### Prompt:
    LLM is told that it is a customer representative from ABC-company and to use chat history and RAG context to answer questions
    If it does not know the answer, it is told to say it does not know and tell user to contact human service
    #### Memory:
    Chat memory is fed into the input so that the chatbot is aware of the context of the conversation.
    However, as the chat history gets long, it becomes confused. It is a limitation of this simple demo.
    #### Temperature: 0.1
    The chatbot is not encouraged to be creative but use factual answers provided in retrieval results.

    #### Good Testing Question Example:
    - Who are you?
        - The answer should show the role assigned in prompt is working.
    - How do I go to Mars?
        - The answer should show that when asked about things it doesn't know or irrelevant, it knows it should refer users to human service.
    - Can I talk to someone? Followed by next query: When can I do that?
        - This question pair should show that the chatbot has memory and it can understand what it means by "that".
    - Other typical customer support questions:
        - Can I request a refund? (or in chinese: 我可以申請退款嗎？)
        - How do I track my order? (or in chinese: 怎樣查找我的訂單？)

    """)

demo.launch()

# demo.close()

#