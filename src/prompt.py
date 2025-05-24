import os
from PIL import Image

dyn_vqa_img1_path = os.path.join('./prompt_img', "dyn_vqa_img1.jpeg")
dyn_vqa_img2_path = os.path.join('./prompt_img', "dyn_vqa_img2.jpg")
dyn_vqa_img3_path = os.path.join('./prompt_img', "dyn_vqa_img3.jpg")


r1_router_prompt = """You are a professional question decomposition expert for multi-hop QA systems. Your task is to decompose complex questions into **strictly single-hop** sub-questions and select appropriate retrievers.

**Strict Output Format:**
<think>
[Analyze the original question and determine the next required sub-question. Do NOT reveal answers or perform multi-hop reasoning.]
</think>
<sub-question>
[Exactly ONE single-hop question one time. If no further information is needed to answer the origin question, write 'None'.]
</sub-question>
<ret> 
[Choose 1 retriever from: Text Retriever, Text Image Retriever, Table Retriever. Write 'None' if <sub-question> is 'None'.]
</ret>

### Critical Rules:
1. **Atomic Sub-question Definition**:
   - A sub-question is "atomic" only if: 
     a) It cannot be further decomposed into simpler questions
     b) It requires exactly **one retrieval action** to answer
     c) Does NOT depend on answers to previous sub-questions
     d) It can be helpful to answer the origin question
   - Example: ❌ "Find the capital and population of France" → ✅ Split into two sub-questions

2. **Retriever Selection Guidelines**:
   - `Text Retriever`: 
     - For non-visual commonsense knowledge (e.g., "Define photosynthesis")
   - `Text Image Retriever`:
     - When sub-question explicitly references visual elements (e.g., "Describe the painting style of...")
   - `Table Retriever`:
     - For numerical/statistical queries (e.g., "GDP of Japan in 2020")

3. **Strict Prohibitions**:
   - Never combine multiple questions in <sub-question>
   - Never mention retrieved content in <think>
   - Never select retrievers for non-atomic questions

### Examples:
"""

r1_router_fewshot_msgs = {
    'hotpotqa': [{'role': 'system', 'content': [
            {"type": 'text', 'text': r1_router_prompt}
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 1: Origin question: Which magazine was started first Arthur's Magazine or First for Women?"}
        ]}, 
        {'role': 'assistant', 'content': [
            {"type": 'text', 'text': """<think>The user wants to compare the starting dates of "Arthur's Magazine" and "First for Women" to see which one was established first.\nWe'll begin by finding out when "Arthur's Magazine" started.\n This information might appear in textual descriptions.</think>\n<sub-question>When did the magazine 'Arthur's Magazine' start?</sub-question>\n<ret>Text Retriever</ret>"""}
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 2: Origin question: Which magazine was started first Arthur's Magazine or First for Women?\nSub-question1: When did the magazine 'Arthur's Magazin' start?\nAnswer1: 'Arthur's Magazin' was started in 2002"}
        ]},
        {'role': 'assistant', 'content': [
            {"type": 'text', 'text': """<think>We already found that "Arthur's Magazine" was started in 2002. To compare which magazine was established first, we now need the start date of "First for Women.\nThis information might appear in textual descriptions</think>\n<sub-question>When did the magazine 'First for Women' start?</sub-question>\n<ret>Text Retriever</ret>"""}
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 3: Origin question: Which magazine was started first Arthur's Magazine or First for Women?\nSub-question1: When did the magazine 'Arthur's Magazin' start?\nAnswer1:'Arthur's Magazin' was started in 2002\nSub-question2: When did the magazine 'First for Women' start?\nAnswer2: 'First for Women' was started in 1880"}
        ]},
        {'role': 'assistant', 'content': [
            {"type": 'text', 'text': """<think>Now we have both start dates: "Arthur's Magazine" (2002) and "First for Women" (1880). \n Clearly, "First for Women" was started earlier, so we have enough information to answer the origin question. </think>\n<sub-question>None</sub-question>\n<ret>None</ret>"""}
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 4: Origin question: What is the population of the city that Manchester is in the southeast corner of?"}
        ]},
        {'role': 'assistant', 'content': [
            {"type": 'text', 'text': "<think>The question asks for the population of the city where Manchester is in its southeast corner.\n To identify that city, we need to check which city is to the northwest of Manchester.\nNames and location details are often found in textual descriptions, so let's first try the Text Retriever.</think>\n<sub-question>which city is in the northwest of Manchester</sub-question>\n<ret>Text Retriever</ret>"}
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 5: Origin question: What is the population of the city that Manchester is in the southeast corner of?\nSub-question1: which city is in the northwest of Manchester?\nAnswer1:It's Bolton"}
        ]},
        {'role': 'assistant', 'content': [
            {"type": 'text', 'text': "<think>We found out that the city in question is Bolton.\nTo answer the origin question about population, we now need the population of Bolton.\nPopulation data might be available in textual descriptions.</think>\n<sub-question>What is the population of Bolton?</sub-question>\n<ret>Text Retriever</ret>"}
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 6: Origin question: What is the population of the city that Manchester is in the southeast corner of?\nSub-question1: which city is in the northwest of Manchester?\nAnswer1:It's Bolton\nSub-question2: What is the population of Bolton?\nAnswer2: It's about 2960K"}
        ]},
        {'role': 'assistant', 'content': [
            {"type": 'text', 'text': "<think>We have identified the city (Bolton) and found its population is about 2960K. That fully answers the origin question about the population of the city in the southeast corner of Manchester.No further retrieval is needed.</think>\n<sub-question>None</sub-question>\n<ret>None</ret>"}
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 7: Origin question: Who directed the film where Joe Pesci voiced the character of Harry Lyme?"}
        ]},
        {'role': 'assistant', 'content': [
            {"type": 'text', 'text': "<think>The user wants to know who directed the film in which Joe Pesci voiced Harry Lyme. First, we need to identify the film itself. Information about Joe Pesci's roles should be in textual references.</think>\n<sub-question>What is the name of the film where Joe Pesci voiced the character of Harry Lyme?</sub-question>\n<ret>Text Retriever</ret>"}
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 8: Origin question: When was the saxophonist who recorded A Love Supreme born?"}
        ]},
        {'role': 'assistant', 'content': [
            {"type": 'text', 'text': """<think>We want the birthdate of the saxophonist who recorded "A Love Supreme."\nFirst, we need to identify the saxophonist. Because this is general knowledge about music history, a text-based reference should suffice.</think>\n<sub-question>Who is the saxophonist who recorded A Love Supreme?</sub-question>\n<ret>Text Retriever</ret>"""}
        ]},
    ],
    'dyn_vqa':  [{'role': 'system', 'content': [
            {"type": 'text', 'text': r1_router_prompt}
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 1: Origin question: which year did he win the world cup champion?"}, 
            {"type": "image", "image": dyn_vqa_img1_path}
        ]},
        {'role': 'assistant', 'content': [
            {"type": 'text', 'text': "<think>We are asked about the year someone won the World Cup. The image might provide a clue, but first we need to identify who this person is. Let's begin by determining the identity of the person in the image.</think></think>\n<sub-question>Who is this person in the image?</sub-question>\n<ret>Text Image Retriever</ret>"},
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 2: Origin question: which year did he win the world cup champion?\nSub-question1: Who is this person in the image?\nAnswer1: He is Leo Messi."},
            {"type": "image", "image": dyn_vqa_img1_path}
        ]}, 
        {'role': 'assistant', 'content': [
            {"type": 'text', 'text': "<think>We have identified the person in the image as Lionel Messi. Since the original question asks about the year he won the World Cup, this information may be available both in textual descriptions. We will use the Text Retriever to gather this information.</think>\n<sub-question>Which year did Lionel Messi win the World Cup champion?</sub-question>\n<ret>Text Retriever</ret>"},
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 3: Origin question: which year did he win the world cup champion?\nSub-question1: Who is this person in the image?\nAnswer1: He is Leo Messi.\nSub-question2: Which year did Lionel Messi win the World Cup champion?\nAnswer2: He won the world cup champion in 2022."},
            {"type": "image", "image": dyn_vqa_img1_path}
        ]},
        {'role': 'assistant', 'content': [
            {"type": "text", "text": "<think>We have already identified that Lionel Messi won the World Cup in 2022, which answers the original question. Therefore, no further retrieval or sub-questions are needed.</think>\n<sub-question>None</sub-question>\n<ret>None</ret>"}
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 4: Origin question: Who is the father of the actor who played the male protagonist in this movie?"},
            {"type": "image", "image": dyn_vqa_img2_path}
        ]},
        {'role': 'assistant', 'content': [
            {"type": "text", "text": "<think>We are asked about the father of the actor who played the male protagonist in this movie. However, before we can determine the actor's father, we first need to identify which movie this is. The image is likely to provide clues, so we will use the Text Image Retriever to gather both image and text data to identify the movie. This approach allows us to handle both visual and textual information effectively.</think>\n<sub-question>What movie is this poster from?</sub-question>\n<ret>Text Image Retriever</ret>"}
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 5: Origin question: Who is the father of the actor who played the male protagonist in this movie?Sub-question1: What movie is this poster from?\nAnswer1: The movie's name is Titanic."},
            {"type": "image", "image": dyn_vqa_img2_path}
        ]},
        {'role': 'assistant', 'content': [
            {"type": "text", "text": "<think>Now that we know the movie is Titanic, we can focus on finding out who played the male protagonist. This is a straightforward textual query, and we will use the Text Retriever to retrieve this information from relevant sources.</think>\n<sub-question>Who played the male protagonist in Titanic?</sub-question>\n<ret>Text Retriever</ret>"}
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 6: Origin question: Who is the father of the actor who played the male protagonist in this movie?Sub-question1: What movie is this poster from?\nAnswer1: The movie's name is Titanic.\nSub-question2: Who played the male protagonist in Titanic?\nAnswer2: He is Leonardo DiCaprio."},
            {"type": "image", "image": dyn_vqa_img2_path}
        ]},
        {'role': 'assistant', 'content': [
            {"type": "text", "text": "<think>We have already identified the movie as Titanic and the actor who played the male protagonist as Leonardo DiCaprio. The next step is to find out who his father is. Since this is factual information, we will now use the Text Retriever to gather information about Leonardo DiCaprio's father.</think>\n<sub-question>Who is the father of Leonardo DiCaprio?</sub-question>\n<ret>Text Retriever</ret>"}
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 7: Origin question: Who is the father of the actor who played the male protagonist in this movie?Sub-question1: What movie is this poster from?\nAnswer1: The movie's name is Titanic.\nSub-question2: Who played the male protagonist in Titanic?\nAnswer2: He is Leonardo DiCaprio.\nSub-question3: Who is the father of Leonardo DiCaprio?\nAnswer3: He is George."},
            {"type": "image", "image": dyn_vqa_img2_path}
        ]},
        {'role': 'assistant', 'content': [
            {"type": "text", "text": "<think>We have already identified the movie as Titanic, the actor as Leonardo DiCaprio, and his father as George. Therefore, the original question about who is the father of the actor has already been answered. No further retrieval or sub-questions are needed.</think>\n<sub-question>None</sub-question>\n<ret>None</ret>"}
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 8: Origin question: where is this man's mentor work in?"},
            {"type": "image", "image": dyn_vqa_img3_path}
        ]},
        {'role': 'assistant', 'content': [
            {"type": "text", "text": "<think>The user is asking about the mentor of the man in the image and where his mentor works. The first step is to identify the man in the image. Once we know who he is, we can proceed to gather information about his mentor. Let's begin by identifying who this man is.</think>\n<sub-question>Who is the person in this image?</sub-question>\n<ret>Text Image Retriever</ret>"}
        ]},
    ],
    'openwikitqa': [{'role': 'system', 'content': [
            {"type": 'text', 'text': r1_router_prompt}
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 1: Origin question: What is the CFL team that has held a position as ol since the 2004 CFL Draft Round 5?"}
        ]}, 
        {'role': 'assistant', 'content': [
            {"type": 'text', 'text': "<think>The question asks about the CFL team and a position held by an OL since the 2004 CFL Draft. This is very likely to be in a table format, such as a record of team drafts and player positions. We will use the Table Retriever to retrieve relevant structured data from sources like the CFL draft tables.</think>\n<sub-question>What CFL team has held a position as OL since the 2004 CFL Draft Round 5?</sub-question>\n<ret>Table Retriever</ret>"}
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 2: Origin question: What is the party with a 51.0% Brooklyn vote in the 1932 New York City mayoral election?"}
        ]},
        {'role': 'assistant', 'content': [
            {"type": 'text', 'text': "<think>This question is asking for a specific political party with a 51% vote share in Brooklyn during the 1932 New York City mayoral election. This type of data is likely available in historical election result tables. We will use Table Retriever to retrieve relevant election data from a structured source.</think>\n<sub-question>Which party received 51.0% of the Brooklyn vote in the 1932 New York City mayoral election?</sub-question>\n<ret>Table Retriever</ret>"}
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 3: Origin question: How many seasons have the Canterbury Bulldogs (8) won in Australian rugby league premieres since 1998 - National Rugby League?"}
        ]},
        {'role': 'assistant', 'content': [
            {"type": 'text', 'text': "<think>The question asks about the number of seasons the Canterbury Bulldogs won in the Australian rugby league since 1998. This data is most likely available in a table, such as National Rugby League championship results. We will use Table Retriever to retrieve the relevant data from structured sources like NRL tables.</think>\n<sub-question>How many seasons have the Canterbury Bulldogs won the Australian rugby league premieres since 1998?</sub-question>\n<ret>Table Retriever</ret>"}
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 4: Origin question: Among all the races of the Grands Prix of the 1992 Formula 1 season, what are all the places where the date is April 5th?"}
        ]},
        {'role': 'assistant', 'content': [
            {"type": 'text', 'text': "<think>The question asks about specific dates during the 1992 Formula 1 season. The details about each race's location and date are typically stored in a table. We will use Table Retriever to look up the relevant data for the 1992 Formula 1 season and find all the races that took place on April 5th.</think>\n<sub-question>What are all the places where the races of the 1992 Formula 1 season took place on April 5th?</sub-question>\n<ret>Table Retriever</ret>"}
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 5: Origin question: When was Erskine College founded for the current members of the Conference Carolinas?"}
        ]}, 
        {'role': 'assistant', 'content': [
            {"type": 'text', 'text': "<think>This question asks about the founding year of Erskine College in relation to its membership in the Conference Carolinas. This is historical data that can be found in  structured records (e.g., college data tables). We will use both Table Retriever to retrieve this information from appropriate sources.</think>\n<sub-question>When was Erskine College founded for the current members of the Conference Carolinas?</sub-question>\n<ret>Table Retriever</ret>"}
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 6: Origin question: What are all the service dates recorded on 21 November 1945 for the Stena Line Holland BV vessels operated by the Zeeland Steamship Company (SMZ)?"}
        ]},
        {'role': 'assistant', 'content': [
            {"type": 'text', 'text': "<think>This question is asking for service dates for specific vessels on a historical date. These kinds of records are typically available in structured historical tables or logs. We will use Table Retriever to get structured data.</think>\n<sub-question>What are all the service dates recorded on 21 November 1945 for the Stena Line Holland BV vessels operated by the Zeeland Steamship Company?</sub-question>\n<ret>Table Retriever</ret>"}
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 7: Origin question: What is the total number of players for the Houston Rockets All-Time W Roster, where the years for Rockets are 2004-06?"}
        ]},
        {'role': 'assistant', 'content': [
            {"type": 'text', 'text': "<think>This question involves the Houston Rockets All-Time W Roster from 2004-06, which is most likely recorded in structured sports data tables, such as player rosters. We will use the Table Retriever to retrieve this information from sports databases.</think>\n<sub-question>What is the total number of players for the Houston Rockets All-Time W Roster, where the years for Rockets are 2004-06?</sub-question>\n<ret>Table Retriever</ret>"}
        ]},
        {'role': 'user', 'content': [
            {"type": 'text', 'text': "Example 8: Origin question: What is the district for 1952 United States House of Representatives elections, 1988, Massachusetts?"}
        ]},
        {'role': 'assistant', 'content': [
            {"type": 'text', 'text': "<think>This question is asking about a specific district for the 1952 United States House of Representatives elections in Massachusetts in 1988. Election data is often found in both structured tables (election results) and textual descriptions. We will use both Table Retriever to retrieve the election results from structured data.</think>\n<sub-question>What is the district for the 1952 United States House of Representatives elections, 1988, Massachusetts?</sub-question>\n<ret>Table Retriever</ret>"}
        ]},
    ],
}

r1_router_fewshot_msgs['infoseek'] = r1_router_fewshot_msgs['dyn_vqa']
r1_router_fewshot_msgs['webqa'] = r1_router_fewshot_msgs['dyn_vqa']
r1_router_fewshot_msgs['2wikimultihopqa'] = r1_router_fewshot_msgs['hotpotqa']
r1_router_fewshot_msgs['tabfact'] = r1_router_fewshot_msgs['openwikitqa']

get_answer_prompt =""""You are a professional question answering model. Your task is to carefully think through the question based on the information retrieved and then provide the final answer.

Strict Output Format: 
<think> 
[Analyze the original question and the retrieved information. Break down the reasoning process step by step. Do NOT provide the final answer yet.] 
</think> 
<answer> 
[Provide the final answer based solely on the retrieved information.]
</answer>
"""

get_final_answer_prompt =""""You are a professional question answering model. Your task is to carefully think through the question based on the sub-quetsions and its answers and then provide the final answer.

Strict Output Format: 
<think> 
[Analyze the original question and sub-questions with its answers. Break down the reasoning process step by step. Do NOT provide the final answer yet.] 
</think> 
<answer> 
[Provide the final answer based solely on the information before.]
</answer>
"""


def convert(msgs):
    qw_prompt = []
    for msg in msgs:
        m = {}

        m['role'] = msg['role']

        m['content'] = ""
        for cont in msg['content']:
            if cont['type'] == "text":
                m['content'] += cont["text"]

        qw_prompt.append(m)

    return qw_prompt