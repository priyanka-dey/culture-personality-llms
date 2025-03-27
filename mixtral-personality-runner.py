from dotenv import load_dotenv
import pandas as pd
import argparse
from pydantic import BaseModel
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from base_personality_prompts import *
import os

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Define the schema for validating responses
class Response(BaseModel):
    answer: str

MODEL_generation = "mixtral-8x7b-32768"

client = ChatGroq(model=MODEL_generation, groq_api_key=groq_api_key)

def call_mixtral(prompt, model, country):
    # messages=[
    #     {"role": "system", "content": f"You are someone who was born and brought up in {country}. You are very familiar with the {country}'s culture and traditions and practice many of them."}, 
    #     {"role": "user", "content": prompt}  
    # ]
    
    system_message = (
        f"You are someone who was born and brought up in {country}. "
        f"You are very familiar with {country}'s culture and traditions and practice many of them."
    )
    human = "{text}"
    
    chat_prompt = ChatPromptTemplate.from_messages([("system", system_message), ("human", human)])
    chain = chat_prompt | client
    response = chain.invoke({"text": prompt})
    result = response.content
    return result

def _process_response(resp, answer_dict={5: 'high', 4: 'mod_high', 3: 'med', 2: 'mod_low', 1: 'low'}):
    start = resp.index('{')
    end = resp.index('}')
    resp_json = resp[start:end+1]
    parsed_resp = Response(**json.loads(resp_json))
    if 'd' in parsed_resp.answer.lower():
        return (answer_dict[5], 5)  # high score
    elif 'b' in parsed_resp.answer.lower():
        return (answer_dict[1], 1)  # low score
    elif 'c' in parsed_resp.answer.lower():
        return (answer_dict[3], 3)  # med score
    elif 'd' in parsed_resp.answer.lower():
        return (answer_dict[2], 2)
    else:
        return (answer_dict[4], 4)  # mod_high score

def _process_TRAIT_response(resp):
    start = resp.index('{')
    end = resp.index('}')
    resp_json = resp[start:end+1]
    parsed_resp = Response(**json.loads(resp_json))
    if 'a' or 'd' in parsed_resp.answer.lower():
        return ('low', 1)
    else:
        return ('high', 5)

def score(traits, scores):
    o = []
    c = []
    e = []
    a = []
    n = []

    for i, t in enumerate(traits):
        if t == 'Openness to Experience':
            o.append(scores[i])
        elif t == 'Conscientiousness':
            c.append(scores[i])
        elif t == 'Extraversion':
            e.append(scores[i])
        elif t == 'Agreeableness':
            a.append(scores[i])
        else:
            n.append(scores[i])

    return {
        'Openness to Experience': sum(o)/len(o),
        'Conscientiousness': sum(c)/len(c),
        'Extraversion': sum(e)/len(e),
        'Agreeableness': sum(a)/len(a),
        'Neuroticism': sum(n)/len(n)
    }

if __name__=='__main__':
    # Load command-line arguments for the personality test.
    parser = argparse.ArgumentParser(prog='PersonalityTester')
    parser.add_argument('-a', '--answers', default='all_USA_answers.csv')
    parser.add_argument('-q', '--questions', default='all_USA_questions.csv')
    parser.add_argument('-t', '--test_type', default="cultureDataset")
    parser.add_argument('-c', '--country', default="USA")
    args = parser.parse_args()

    questions = pd.read_csv(f"./cultureDataset/{args.questions}")
    answers = pd.read_csv(f"./cultureDataset/{args.answers}")

    if args.test_type == 'cultureDataset':
        answers['scenario'] = questions['scenario_text'].values
        model_responses = []
        model_scores = []
        traits = []
        for i, row in answers.iterrows():
            ques = row['question']
            high = row['high']
            mod_high = row['moderately_high']
            med = row['medium']
            low = row['low']
            mod_low = row['moderately_low']
            s = row['scenario']
            t = row['trait']
            prompt = culture_prompt.format(
                scenario=s, question=ques, high=high, mod_high=mod_high, med=med, mod_low=mod_low, low=low,
            )

            print(f"Prompting model for {args.country} cultureDataset question {i} *****************************")

            response = call_mixtral(prompt, MODEL_generation, args.country)
            resp, score_val = _process_response(response)
            model_responses.append(resp)
            model_scores.append(score_val)
            traits.append(t)

            print(f"Parsed model response for {args.country} question {i} *****************************")

        answers[f'mixtral_{args.test_type}_responses'] = model_responses
        answers[f'mixtral_{args.test_type}_scores'] = model_scores

        answers.to_csv(f"{args.country}_{args.questions}_mixtral_prompted_personality.csv", index=False)

    elif args.test_type == 'standard':
        model_responses = []
        model_scores = []
        traits = []
        for i, row in questions.iterrows():
            q = row['Text'].lower()
            reverse = row['Reverse']
            prompt = standard_prompt.format(question=q)
            print(f"Prompting model for {args.country} {args.questions} question {i} ************************")
            response = call_mixtral(prompt, MODEL_generation, args.country)
            if reverse == 'F':
                resp, score_val = _process_response(response, answer_dict={
                    1: 'Very inaccurate', 2: 'Moderately inccurate', 3: 'Neither accurate nor inaccurate',
                    4: 'Moderately accurate', 5: 'Very accurate'
                })
                print(f"Parsed model response (forward) for {args.country} {args.questions} question {i} ************************")
            else:
                resp, score_val = _process_response(response, answer_dict={
                    5: 'Very inaccurate', 4: 'Moderately inccurate', 3: 'Neither accurate nor inaccurate',
                    2: 'Moderately accurate', 1: 'Very accurate'
                })
                print(f"Parsed model response (reverse) for {args.country} {args.questions} question {i} ************************")

            t = row['Key']
            model_responses.append(resp)
            model_scores.append(score_val)
            traits.append(t)

        questions[f'mixtral_{args.questions}_responses'] = model_responses
        questions[f'mixtral_{args.questions}_scores'] = model_scores

        question_type = args.questions.split("/")[1]
        questions.to_csv(f"{args.country}_{question_type}_mixtral_prompted_personality.csv", index=False)

    else:
        # TRAIT dataset processing
        model_responses = []
        model_scores = []
        traits = []
        for i, row in questions.iterrows():
            ques = row['question']
            high1 = row['high1']
            high2 = row['high2']
            low1 = row['low1']
            low2 = row['low2']
            s = row['situation']
            t = row['trait']
            prompt = standard_prompt.format(scenario=s, question=ques)
            print(f"Prompting model for {args.country} TRAIT question {i} ************************")
            response = call_mixtral(prompt, MODEL_generation, args.country)
            resp, score_val = _process_TRAIT_response(response)
            print(f"Parsed model response (reverse) for TRAIT {args.questions} question {i} ************************")

            model_responses.append(resp)
            model_scores.append(score_val)
            traits.append(t)

        questions[f'mixtral_{args.questions}_responses'] = model_responses
        questions[f'mixtral_{args.questions}_scores'] = model_scores

        questions.to_csv(f"{args.country}_{args.questions}_mixtral_prompted_personality.csv", index=False)
