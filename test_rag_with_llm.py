from openai import OpenAI
import os
from pinecone import Pinecone
import json
pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

project_name = "atv_reports_v2"
question_path = f"{project_name}/data/transcription_metadata.json"
report_template_path = f"{project_name}/data/pqr_template.md"
pqr_report_template = open(report_template_path).read()

questions = json.load(open(question_path))





cleaned_project_name = project_name.replace('_','-')
index_name = f"{cleaned_project_name}-transcripts"
name_space = "all_text_transcripts_page_embeddings"

instructions = """You are a helpful assistant that can answer questions about the product description. 
                                                      You are given a question and a product description.
                                                      You need to answer the question based on the product description.
                                                      You need to extract the information from the product description and return it in a structured """

pqr_report_creation_instruction = """Based on the output template and context. Fill the markdown template with all the relevant information"""



dense_index = pc.Index(index_name)

def get_context(question,dense_index,name_space):
# Define the query
# Search the dense index
    results = dense_index.search(
        namespace=name_space,
        query={
            "top_k": 15,
            "inputs": {
                'text': question
            }
        }
    )
    return results

# Print the results
all_context = []
output_json = {}
for question_key in questions:
    question = str(questions[question_key])
    print(question)
    results = get_context(question,dense_index,name_space)
    for hit in results['result']['hits']:
            all_context.append(hit['fields']['chunk_text'])



    input_context = "\n".join(all_context)

    input_context = f"""instructions: {instructions}
                        input_context: {input_context}
                        Here is the question: {question}"""
    response = client.responses.create(input = input_context,
                                        model = "gpt-4o")

    output_json[question_key] = response.output_text

pqr_report_prompt = f"""{pqr_report_creation_instruction}.
                    Output json : {output_json}
                    output template : {pqr_report_template}
                    """


response = client.responses.create(input = pqr_report_prompt,
                                    model = "gpt-4o")

pqr_report = response.output_text

with open(f"{project_name}/data/pqr_report.md", "w") as f:
    f.write(pqr_report)
    f.close()