culture_prompt = """
    Please read the following scenario and question carefully and choose how you would respond in the scenario. 
        Scenario: {scenario} 
        Question: {question} 
        Answers:             
            a: {mod_high}
            b: {low} 
            c: {med}
            d: {high}
            e: {mod_low}
        Please choose from: a, b, c, d, e.
        Please generate the response as a **valid JSON object** with the following key: 'answer'.
"""

standard_prompt = """
    Please read the following statement carefully and please choose which answer best describes you.
    Question: I {question}
    Answers: 
        a: Very inaccurate 
        b: Moderately inaccurate 
        c: Neither accurate nor inaccurate 
        d: Moderately accurate 
        e: Very accurate
    Please choose from: a, b, c, d, e. 
    Please generate the response as a **valid JSON object** with the following key: 'answer'.
"""

trait_prompt = """
        Please read the following scenario and question carefully and choose how you would respond in the scenario. 
        Scenario: {scenario} 
        Question: {question} 
        Answers:             
            a: {low_1}
            b: {high_2} 
            c: {high_1}
            d: {low_2}
        Please choose from: a, b, c, d.
        Please generate the response as a **valid JSON object** with the following key: 'answer'.

"""