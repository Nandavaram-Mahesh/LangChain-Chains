import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

google_gemma_llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=google_gemma_llm)   

parser = StrOutputParser()

# prompt1 = PromptTemplate(
#     template='Classify the sentiment of the following feedback text into postive or negative \n {feedback}',
#     input_variables=['feedback'],
#     # partial_variables={'format_instruction':parser.get_format_instructions()}
# )

# classifier_chain = prompt1 | model | parser

# print(classifier_chain.invoke({'feedback':'I am not sure about this product'}))


# The sentiment of the feedback text "The product is very good" is **positive**.

# Here's why:

# * **"very good"** is a clear indication of a positive opinion about the product.

# There's nothing in the text to suggest anything negative



# The sentiment of the feedback text is **neutral**.

# Here's why:

# * **"Not sure"** expresses uncertainty or a lack of confidence. This is not a strongly positive or negative statement.
# * It's missing an actual positive or negative opinion about the product. The person may not have fully formed feelings yet.


# While the feedback is neutral, it implicitly suggests a lack of confidence and could potentially lead to more negative information.



# As we can see the result can be either positive or negative or anything else ... , there is no strucuted output
# so to get structured output we need to use PydanticOutput parser

class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')


parser2 = PydanticOutputParser(pydantic_object=Feedback)


prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

# print(classifier_chain.invoke({'feedback':'This is a great product'}).sentiment)

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)


branch_chain = RunnableBranch(
    # (Condition,chain)
    # classifier_chain = prompt1 | model | parser2
    # x is the result of this classifier_chain.invoke({'feedback':'This is a great product'})
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    # If no condition matches, use Default condition
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'This is a beautiful phone'}))

# <think>

# Okay, the user wants me to craft a response to positive feedback. They've explicitly stated the sentiment is positive, so my response should be warm and appreciative.

# Hmm, the user didn't provide the actual feedback content, which makes this interesting. That means I need to create a template-style response that works for any positive comment. The challenge is making it feel genuine without being too generic.

# I should focus on three key elements: gratitude, validation of their effort, and future commitment. People giving positive feedback usually want to feel heard and know their appreciation matters. Maybe they're also subtly checking if the praised behavior will continue?

# Right, tone is crucial here. Too formal feels robotic, too casual might seem insincere. That corporate "delighted" phrase strikes a good balance - professional but human. Adding the "made our day" bit personalizes it.

# The second sentence does important work: "comments like yours" implies they're part of a valued group, "why we do what we do" connects their feedback to our purpose. That's psychology 101 - positive reinforcement loop.

# Ending with the "looking forward" bit is strategic. It turns their praise into future commitment while feeling proactive. No empty "we'll try harder" vagueness. Smart.

# Wait - should I offer a call-to-action? No, unsolicited CTAs after praise can feel transactional. Unless they asked a question in their feedback (which we don't know), best to keep it pure gratitude.

# *checks response length* Four sentences is perfect. Enough to feel substantial, not so much that it drowns the compliment. The emoji is risky but 🤩 works - it's exuberant but not unprofessional in modern comms.

# User probably needs this for business context. If it was personal feedback, I'd drop the "team" references. But since they didn't specify, B2B tone is safer default.</think>


# Here's a warm and professional response to positive feedback:

# **"Thank you so much for your wonderful feedback! We're truly delighted to hear you had such a positive experience. Comments like yours are exactly why we do what we do – it makes our entire team's day! We look forward to continuing to exceed your expectations."**

# ### Why this works:
# 1. **Immediate Gratitude** ("Thank you so much")
# 2. **Mirror Sentiment** ("delighted," "positive experience")
# 3. **Team Recognition** ("entire team's day")
# 4. **Future Commitment** ("continue to exceed expectations")
# 5. **Natural Enthusiasm** (Exclamation points without overdoing it)

# **Alternative options:**
# - For brief settings: *"This means the world to us! Thanks for cheering us on – we’ll keep raising the bar!"*
# - For personal service: *"I’m thrilled this resonated with you! It was a pleasure helping, and I’m always here for you."*

# Include a 🤩 or 😊 emoji if your brand voice allows! Always match the tone to your relationship with the reviewer.


chain.get_graph().print_ascii()



#     +-------------+      
#     | PromptInput |
#     +-------------+
#             *
#             *
#             *
#    +----------------+
#    | PromptTemplate |
#    +----------------+
#             *
#             *
#             *
#   +-----------------+
#   | ChatHuggingFace |
#   +-----------------+
#             *
#             *
#             *
# +----------------------+
# | PydanticOutputParser |
# +----------------------+
#             *
#             *
#             *
#        +--------+
#        | Branch |
#        +--------+
#             *
#             *
#             *
#     +--------------+
#     | BranchOutput |
#     +--------------+
