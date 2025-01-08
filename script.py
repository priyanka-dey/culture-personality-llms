from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import json
import argparse

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["HUGGING_FACE_API_KEY"] = os.getenv("HUGGING_FACE_API_KEY")
os.environ["OPEN_AI_API"] = os.getenv("OPEN_AI_API")
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
open_ai_api_key = os.getenv("OPEN_AI_API")

documents = list()

def load_documents(root_directory):
    for folder, _, files in os.walk(root_directory):
        for file in files:
            file_path = os.path.join(folder, file)
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")


# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=200)
chunked_documents = []

for doc in tqdm(documents, desc="Splitting Documents"):
    chunks = text_splitter.split_documents([doc])
    chunked_documents.extend(chunks)
    
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=open_ai_api_key)
vector_db = FAISS.from_documents(chunked_documents, embedding_model)
llm = ChatGroq(model='llama-3.2-3b-preview', groq_api_key=groq_api_key)

# Memory setup
memory = ConversationBufferMemory(
    memory_key="chat_history",  # Key used to store conversation history
    return_messages=True        # Allows memory to be added to chain responses
)

# Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
    memory=memory,
    chain_type="stuff"
)

def generate_cultural_norms(country: str) -> dict:
    prompt = f"Provide me 10 unique sentences highlighting the core values/important aspects of individuals living in {country}."
    response = qa_chain.run(prompt)
    print(response)
    # Split the response into lines and filter undesired content
    norms = []
    for idx, line in enumerate(response.split("\n")):
        line = line.strip()
        # Check if the line is valid (e.g., non-empty and not generic text)
        if line and not line.lower().startswith("i don't know") and not "here are" in line.lower():
            # Optionally, check for a pattern (e.g., starting with a number or bullet)
            if line[0].isdigit() or line.startswith("-"):
                norms.append({"id": len(norms) + 1, "text": line})
    
    # If no valid norms are found, return an empty list
    if not norms:
        print(f"No valid norms generated for {country}.")
        return {"gpt4-o prompt": prompt, "country": country, "norms": []}
    
    return {
        "gpt4-o prompt": prompt,
        "country": country,
        "norms": norms
    }


def generate_cultural_scenarios(country: str, norms: list, batch_size: int = 3) -> list:
    scenarios_list = []
    scenario_id = 1

    for i in range(0, len(norms), batch_size):
        batch = norms[i:i + batch_size]
        for norm in batch:
            # Construct the prompt for each norm
            prompt = (
                f"I would like to generate some example scenarios showing these cultural norms in {country}:\n"
                + "\n".join([f"{idx + 1}. {norm['text']}" for idx, norm in enumerate(batch)])
                + "\nPlease generate 10 scenarios for each norm, detailing each scenario with up to 2 sentences. "
                "Please refrain from stating the cultural norm in the scenario."
            )

            try:
                # Run the prompt and handle API response
                response = qa_chain.run(prompt)
                # Process response: Split into lines and filter relevant content
                scenario_lines = [
                    line.strip()
                    for line in response.split("\n")
                    if line.strip() and not line.startswith("Here are") and not line.startswith("**")
                ]
                # Structure scenarios for the given norm
                structured_scenarios = {
                    "gpt4-o prompt": prompt,
                    "country": country,
                    "scenarios": [
                        {
                            "id": scenario_id + idx,
                            "norm-id": norm['id'],
                            "text": scenario
                        }
                        for idx, scenario in enumerate(scenario_lines[:10], start=1)
                    ]
                }

                # Append to the list
                scenarios_list.append(structured_scenarios)
                scenario_id += 10
            except Exception as e:
                print(f"Error generating scenarios for norm '{norm['text']}': {e}")
                continue

    return scenarios_list

# Save or append to JSON file
def save_to_json(data: dict, filename: str):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        if "cultural-norms" in data:
            existing_data.setdefault("cultural-norms", []).extend(data["cultural-norms"])
        if "cultural_scenarios" in data:
            existing_data.setdefault("cultural_scenarios", []).extend(data["cultural_scenarios"])
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)
    else:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
            
# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Country for which to generate cultural norms and scenarios")
    parser.add_argument("country", help="The name of the country.", type=str)
    
    args = parser.parse_args()
    country = args.country

    root_directory = ".\Documents"
    load_documents(root_directory)

    # Generate cultural norms
    cultural_norms = generate_cultural_norms(country)
    save_to_json({"cultural-norms": [cultural_norms]}, "rag_cultural_norms.json")

    # Generate cultural scenarios
    cultural_scenarios = generate_cultural_scenarios(country, cultural_norms['norms'])
    save_to_json({"cultural_scenarios": cultural_scenarios}, "rag_cultural_scenarios.json")

    print(f"Cultural norms and scenarios for {country} saved successfully.")