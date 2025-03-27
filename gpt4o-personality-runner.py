# baseline personality tests: use llama + gpt4o + 2 other models 

import pandas as pd 
import argparse 
from openai import OpenAI
from base_personality_prompts import *
from pydantic import BaseModel 
import json 

# Define the schema
class Response(BaseModel):
    answer: str

api_key = "OPEN_AI_API_KEY"
MODEL_generation = "gpt-4o-mini"
client = OpenAI(api_key = api_key)

# this will need to change to llama and other models 
def call_gpt(prompt, model, country):
    
    completion_model = client.chat.completions.create(
    model = model,
    stream = False,
    messages=[
        {"role": "system", "content": f"You are someone who was born and brought up in {country}. You are very familiar with the {country}'s culture and traditions and practice many of them."}, 
        {"role": "user", "content": prompt}  
    ]
    )

    result = completion_model.choices[0].message.content
    return result

def _process_response(resp, answer_dict = {5: 'high', 4: 'mod_high', 3: 'med', 2: 'mod_low', 1: 'low'}):
    start = resp.index('{')
    end = resp.index('}')
    resp = resp[start:end+1]
    resp = Response(**json.loads(resp)) 
    if 'd' in resp.answer.lower():
        return (answer_dict[5], 5) # high score 
    elif 'b' in resp.answer.lower(): 
        return (answer_dict[1], 1) # low score 
    elif 'c' in resp.answer.lower(): 
        return (answer_dict[3], 3) # med score 
    elif 'd' in resp.answer.lower(): 
        return (answer_dict[2], 2)
    else: 
        return (answer_dict[4], 4) # mod_high score
    
def _process_TRAIT_response(resp):
    start = resp.index('{')
    end = resp.index('}')
    resp = resp[start:end+1]
    resp = Response(**json.loads(resp)) 
    if 'a' or 'd' in resp.answer.lower(): 
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
    # first load the country & norms file 
    parser = argparse.ArgumentParser(
                    prog='PersonalityTester',
                    )        
    parser.add_argument('-a', '--answers', default='all_USA_answers.csv')
    parser.add_argument('-q', '--questions', default='all_USA_questions.csv')
    parser.add_argument('-t', '--test_type', default="cultureDataset")
    parser.add_argument('-c', '--country', default="USA")
    args = parser.parse_args()

    # questions = pd.read_csv(f"./cultureDataset/{args.questions}")
    # answers = pd.read_csv(f"./cultureDataset/{args.answers}")    

    if args.test_type == 'cultureDataset':
        questions = pd.read_csv(f"./cultureDataset/{args.questions}")
        answers = pd.read_csv(f"./cultureDataset/{args.answers}")
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
                scenario=s, question=ques, high=high, mod_high=mod_high, med=med, mod_low = mod_low, low=low,
            )

            print(f"Prompting model for {args.country} cultureDataset question {i} *****************************")

            response = call_gpt(prompt, MODEL_generation, args.country)
            resp, score = _process_response(response)
            model_responses.append(resp)
            model_scores.append(score) 
            traits.append(t) 

            print(f"Parsed model response for {args.country} question {i} *****************************")

        answers[f'gpt4o-mini_{args.test_type}_responses'] = model_responses
        answers[f'gpt4o-mini_{args.test_type}_scores'] = model_scores

        answers.to_csv(f"{args.country}_{args.questions}_culturedataset_gpt40_prompted_personality.csv", index=False)

    elif args.test_type == 'standard':
        questions = pd.read_csv("./psychometric_tests/ipip-scoring-tools/ipip-120.csv")
        # fill in the prompt with the ipip 120 questions 
        model_responses = [] 
        model_scores = [] 
        traits = [] 
        for i, row in questions.iterrows(): 
            q = row['Text'].lower()
            reverse = row['Reverse']
            prompt = standard_prompt.format(question=q)
            print(f"Prompting model for {args.country} {args.questions} question {i} ************************")
            response = call_gpt(prompt, MODEL_generation, args.country)
            if reverse == 'F':
                resp, score = _process_response(response, answer_dict={
                    1: 'Very inaccurate', 2: 'Moderately inccurate', 3: 'Neither accurate nor inaccurate',
                    4: 'Moderately accurate', 5: 'Very accurate'
                })
                
                print(f"Parsed model response (forward) for {args.country} {args.questions} question {i} ************************")

            else: 
                resp, score = _process_response(response, answer_dict={
                    5: 'Very inaccurate', 4: 'Moderately inccurate', 3: 'Neither accurate nor inaccurate',
                    2: 'Moderately accurate', 1: 'Very accurate'
                })

                print(f"Parsed model response (reverse) for {args.country} {args.questions} question {i} ************************")

            t = row['Key']
            model_responses.append(resp)
            model_scores.append(score) 
            traits.append(t) 

        questions[f'gpt4o-mini_{args.questions}_responses'] = model_responses
        questions[f'gpt4o-mini_{args.questions}_scores'] = model_scores

        # question_type = args.questions.split("/")[1]

        questions.to_csv(f"{args.country}_standard_gpt4o_prompted_personality.csv", index=False)
        # questions.to_csv(f"{args.country}_{question_type}_standard_gpt4o_prompted_personality.csv", index=False)

    else: 
        questions = pd.read_csv('./psychometric_tests/TRAIT_questions.csv')

        # TRAIT dataset 
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
            response = call_gpt(prompt, MODEL_generation, args.country)
            resp, score = _process_TRAIT_response(response)
            
            print(f"Parsed model response (reverse) for TRAIT {args.questions} question {i} ************************")

            model_responses.append(resp)
            model_scores.append(score) 
            traits.append(t) 
            
        questions[f'gpt4o-mini_{args.questions}_responses'] = model_responses
        questions[f'gpt4o-mini_{args.questions}_scores'] = model_scores

        questions.to_csv(f"{args.country}_{args.questions}_trait_gpt4o_prompted_personality.csv", index=False)


# python gpt4o-personality-runner.py -a all_USA_answers.csv -q all_USA_questions.csv -t standard -c USA