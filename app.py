import math
import os, tempfile

import streamlit as st
from dotenv import load_dotenv
import re
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from collections import Counter
from itertools import repeat, chain
import spacy
import openai


st.set_page_config(
    page_title="QA with iMIS EMS",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://help.imis.com/enterprise/',
        'Report a bug': "https://help.imis.com/",
        'About': "Proof of concept AI powered question answering app using the iMIS EMS documentation"
    }
)

st.markdown(
    """
<style>
.css-10oheav {
    padding-top: 18px;
}

.css-vk3wp9 {
    width: 386px !important;
}

footer {visibility: hidden;}

</style>
""",
    unsafe_allow_html=True,
)

if check_password():
    startup()
    write_sidebar()
    run_prompt()


def startup():

    print("starting up ...")
    persist_directory = "./chromadb8"
    collection_name = "imis_docs"
    
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_KEY')
    
    print("load client ...")
    
    
    nlp = get_NLP()
    embedding_function = get_embredding_function()
    client = get_client()
    
    print("client loaded...")
    # client.persist()


@st.cache_resource
def get_client():
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    return chroma_client


@st.cache_resource
def get_NLP():
    nlp = spacy.load("en_core_web_sm")
    return nlp


@st.cache_resource
def get_embredding_function():
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    #default_ef = embedding_functions.DefaultEmbeddingFunction()
    return embedding_function


def write_sidebar():
    with st.sidebar:
        st.title('QA with iMIS EMS')
        st.write('Proof of concept AI powered question answering app using the iMIS EMS documentation.')
    
        st.write('This is a Retrieval Augmented Generation (RAG) system with a local vector store of '
                 'content from https://help.imis.com/enterprise/. The vector store is returns sematic '
                 'matches for questions given. These matches are filtered, combined and sent to a LLM to generate '
                 'the final answer.')
        st.write('This app does not have a memory, each question is answered independently. Ask a question about iMIS EMS '
                 'functionality and hit Find Answer (or enter)...')
    
        answer_words = st.slider('Answer word length', 50, 250, 150, step=10)
        answer_style = st.radio("Answer style", ["Friendly", "Direct", "Professional"])
    
    prompt = st.text_input("Ask me a question about iMIS EMS")
    run_search = st.button("Find Answer")
    
    if 'question' not in st.session_state:
        st.session_state['question'] = ''



@st.cache_resource
def get_collection():
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)
    return collection


def create_chromadb_from_docs(docs):
    db = Chroma.from_documents(docs, mbedding_function=embedding_function, collection_name=collection_name,
                               persist_directory=persist_directory)
    db.persist()


def sort_by_most_common(ini_list):
    new_list = list(chain.from_iterable(repeat(i, c)
                                        for i, c in Counter(ini_list).most_common()))
    return new_list


def get_single_answer_from_text(question, text, words):
    knowledge_system = get_knowledge_system(text)
    complete_question = get_complete_question(question, words)

    response = get_openai_response(system=knowledge_system, question=complete_question, temp=1)

    return response


def original_question(question):
    question_template = get_original_question(question)

    # print("original question: " + question)

    question_reword = get_openai_response(system="", question=question_template, temp=1)
    print("fixed question: " + question_reword)

    if question_reword != "ERROR":
        return question_reword
    else:
        return question


def reword_question(question):
    reword_system_template = get_reword_system()
    reword_question_template = get_reword_question(question)
    # print("original question: " + question)

    question_reword = get_openai_response(system=reword_system_template, question=reword_question_template, temp=1)
    # print("reword question: " + question_reword)

    if question_reword != "ERROR":
        return question_reword
    else:
        return question


def search_collection(question):
    with st.spinner("Searching embeddings... "):

        collection = get_collection()

        # question = original_question(question)

        query_question = reword_question(question)

    with st.spinner("Calculating semantic distances... "):

        question = extract_original_question(query_question)

    with st.spinner("Extracting texts ... "):

        question_versions = extract_question_versions(query_question)

        print("original_question: " + question)
        print("question_versions: " + question_versions)
        # print("three questions: " + three_questions)
        print("search vector database")

        input_em = embedding_function([question_versions])

        results = collection.query(
            query_embeddings=input_em,
            # query_texts=query,
            n_results=24
            # where={"metadata_field": "is_equal_to_this"},
            # where_document={"$contains": "search_string"}
        )

        # print(results["documents"][0])
        print(results["metadatas"][0])

        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]

        # print(distances)
        answer_labels = []
        file_names = []
        line_names = []
        answers = []

    with st.spinner("Combining texts ... "):

        i = 0
        for metas in metadatas:
            metasid = str(metas['id'])
            filename = metasid.split(".")[0]
            line = str(metas['line'])
            file_names.append(filename + ".txt")
            line_names.append(line)
            answer_labels.append(metas['label'])
            answer = {"Label": metas['label'], "Link": metas['link'], "Section": metas['section'], "Distance": distances[i], "File": filename,
                      "Line": line}
            answers.append(answer)
            # print(answer)
            # print(documents[i])
            i += 1

        top3distance = []
        top3distance.append(answers[0])
        top3distance.append(answers[1])
        top3distance.append(answers[2])

        sorted_labels = sort_by_most_common(answer_labels)
        unique_labels = list(dict.fromkeys(sorted_labels))

        sorted_files = sort_by_most_common(file_names)
        unique_files = list(dict.fromkeys(sorted_files))

        print("unique results: " + str(len(unique_labels)))

        articles = []

        i = 0
        for label in unique_labels:
            this_count = Counter(answer_labels)[label]
            filename = ""
            for answer in answers:
                if answer["Label"] == label:
                    filename = answer["File"]
                    break

            articles.append({"Label": label, "Link": "", "Count": this_count, "Dist": "", "File": filename, "Line": ""})
            i += 1

        for answer in answers:
            for article in articles:
                if article["Label"] == answer["Label"]:
                    article["Dist"] += str(answer["Distance"]) + " "
                    article["Line"] += answer['Line'] + " "
                    article["Link"] = answer['Link']

        top3counts = []
        top3counts.append(articles[0])
        top3counts.append(articles[1])
        top3counts.append(articles[2])

        for top3 in top3counts:
            print(top3)

        top_answers = []
        top_labels = []

        for answer in top3counts:
            if answer["Label"] not in top_labels:
                top_answers.append(answer)
                top_labels.append(answer["Label"])

        for answer in top3distance:
            if answer["Label"] not in top_labels:
                top_answers.append(answer)
                top_labels.append(answer["Label"])

        texts = get_texts_from_answers(top_answers)

        full_template = "The following articles are from the iMIS EMS documentation. \n"

    with st.spinner("Generating template  ... "):

        print('number of texts: ' + str(len(texts)))

        if len(texts) == 7:
            limits = [600, 400, 400, 400, 300, 300, 300]
        elif len(texts) == 6:
            limits = [600, 500, 400, 400, 300, 300]
        elif len(texts) == 5:
            limits = [700, 600, 400, 400, 400]
        elif len(texts) == 4:
            limits = [800, 600, 500, 500]
        elif len(texts) == 3:
            limits = [800, 600, 600]
        elif len(texts) == 2:
            limits = [1200, 800]
        elif len(texts) == 1:
            limits = [2000]

        i = 0

        print('number of limits: ' + str(len(limits)))
        toplinks = []
        toplabels = []

        for text in texts:
            # print("******************************************************************************************************")
            answer = top_answers[i]
            print(answer)
            linemeta = str(answer["Line"]).rstrip()
            distmeta = str(answer["Line"]).rstrip()
            print("answer line: " + str(answer["Line"]))

            lines = linemeta.split(" ")
            distances = distmeta.split(" ")
            wordcount = get_word_count(text)

            # print("word count: " + str(wordcount))

            print(str(top_answers[i]['Label']))
            toplinks.append(top_answers[i]['Link'])
            toplabels.append(top_answers[i]['Label'])

            if wordcount > limits[i]:
                text = get_max_text(lines, wordcount, text, distances, limits[i])

            full_template += "From Article: " + str(top_answers[i]['Label']) + "\n"
            full_template += text + "\n"
            full_template += "--------------------------------------------------\n"

            i += 1

        print("FULL TEMPLATE: \n" + full_template)
        print("********************************************************************")

        #final_question_template = get_final_question(question)
        final_question_template = get_helpme_question(question)
        print("QUESTION TEMPLATE: \n" + final_question_template)
        print("********************************************************************")

    with st.spinner("Finding your answer... "):
        question_response = get_openai_response(system=full_template, question=final_question_template, temp=1)

        print("FINAL ANSWER: " + question_response)
        print("********************************************************************")

    additional = "<hr>" \
                 "<p>" \
                 "I don't always get it right. This answer comes from reading the following pages, please " \
                 "read them yourself, or contact the iMIS support team to make sure you have the right information." \
                 "<br>Sources:"

    i = 0
    for link in toplinks:
        additional += "<br><a href='https://help.imis.com/enterprise/" + link + "' target='_blank'>" + toplabels[i] + "</a>"
        i += 1
    
    additional += "</p>"
    
    st.write(question_response)
    st.write(additional, unsafe_allow_html=True)

    return question_response


def extract_answer(answer_text):
    answer_text = str(answer_text)
    answer_pos = answer_text.find("ANSWER:")
    answer_start = answer_text.find(":", answer_pos) + 2
    text_needed = answer_text[answer_start:]

    return text_needed


def extract_question_versions(answer_text):
    answer_text = str(answer_text)
    answer_pos = answer_text.find("VERSIONS:")
    answer_start = answer_text.find(":", answer_pos) + 2
    text_needed = answer_text[answer_start:]

    return text_needed


def extract_original_question(answer_text):
    answer_text = str(answer_text)

    orig_pos = answer_text.find("ORIGINAL:")
    answer_start = answer_text.find(":", orig_pos) + 2

    ver_pos = answer_text.find("VERSIONS:")
    text_needed = answer_text[answer_start:ver_pos]

    return text_needed


def get_max_text(lines, wordcount, text, distances, limit):
    # we want approx 500 words.
    # get average words per line
    linenumbers = []

    for line in lines:
        linenumbers.append(int(line))

    maxline = max(linenumbers)
    minline = min(linenumbers)

    print("limit: " + str(limit))

    print("min line: " + str(minline))
    print("max line: " + str(maxline))

    textlines = text.splitlines(True)
    linecount = len(textlines)
    print("line count: " + str(len(textlines)))

    avgwords = math.floor(wordcount / linecount)

    linesneeded = math.floor(limit / avgwords)
    print("lines needed: " + str(linesneeded))

    if linesneeded > linecount:
        return text
    else:
        line_gap = maxline - minline
        print("line gap: " + str(line_gap))

        if line_gap > linesneeded:
            distnumbers = []

            for dist in distances:
                distnumbers.append(float(dist))

            shortest_distance = min(distnumbers)

            i = distnumbers.index(shortest_distance)
            minline = linenumbers[i]
            maxline = minline
            line_gap = 0

        extras_needed = linesneeded - line_gap
        print("extra lines needed: " + str(extras_needed))

        lines_before = math.floor(extras_needed / 4)

        print("lines_before: " + str(lines_before))

        if (minline - lines_before < 0):
            lines_before = 0
            minline = 0
        else:
            minline = minline - lines_before

        lines_after = extras_needed - lines_before
        print("lines_after: " + str(lines_after))

        if maxline + lines_after >= linecount:
            maxline = linecount - 1
            minline = maxline - linesneeded
        else:
            maxline = maxline + lines_after

        print("minline: " + str(minline))
        print("maxline: " + str(maxline))

        text = get_text_lines(minline, maxline, textlines)

    return text


def get_text_lines(start_line, end_line, lines):
    i = 0
    returntext = ""
    for line in lines:
        if i <= end_line and i >= start_line:
            returntext += str(line)
        i += 1

    return returntext


def get_word_count(text):
    # Load the English language model

    text = text.replace("	", "")
    text = re.sub(r'[^\x00-\x7F]', '', text)
    lines = text.splitlines()
    text = ' '.join(lines)
    text = re.sub(" +", " ", text)

    doc = nlp(text)
    word_count = len(doc)

    return word_count


def get_texts_from_answers(top_answers):
    texts = []

    for answer in top_answers:
        filename = str(answer["File"]) + ".txt"
        filepath = f"imistext/{filename}"

        check_is_file = os.path.isfile(filepath)
        if check_is_file:
            with open(filepath, encoding="utf-8") as f:
                text = f.read()
                texts.append(text)

    return texts


def get_knowledge_system(content):
    template = f"The following content is from the iMIS EMS user documentation. \n" \
               f"### CONTENT ### \n" \
               f"{content} \n"
    return template


def get_rating_system(content, questions):
    template = f"### INSTRUCTIONS ### \n" \
               f"The CONTENT below is from the iMIS EMS user documentation. Your job is to determine if it can be used " \
               f"to answer the question 'What is the 'Sell on Web' Option for products in iMIS?'To figure out if the " \
               f"content is useful, first answer the following 'three additional questions':\n " \
               f"{questions} \n" \
               f"Once you have determined the contents usefulness use the following rules to give a RATING to your " \
               f"answer:\n" \
               f"COMPLETE: all three addition question are answered YES\n" \
               f"PARTIAL: only one or two of the additional questions are answered YES\n" \
               f"NONE: None of the additional questions are answered YES.\n" \
               f"### CONTENT ### \n" \
               f"{content} \n"
    return template


def get_rating_question(question):
    template = f"ORIGINAL QUESTION: '{question}' Provide the answer " \
               f"using the following template" \
               f"### TEMPLATE ###\n" \
               f"RATING: Rating goes here. eg PARTIAL if the answer is useful.\n" \
               f"REASON: Answer each of the 'three additional questions here'.\n" \
               f"ANSWER: Write a helpful and direct answer to the ORIGINAL QUESTION here. Aim the answer at " \
               f"an iMIS EMS user. Don't talk about 'the content' just answer the question.  Maximum 120 words"

    return template


def get_rating_question2(question):
    template = f"Carefully read the CONTENT provided. Use only the text from the CONTENT to write a helpful and " \
               f"accurate answer to the following question from an iMIS user. \n\n" \
               f"QUESTION: {question} \n\n" \
               f"Use the following rules to give a RATING to the answer: \n" \
               f"COMPLETE: The answer is comprehensive.\n" \
               f"GOOD: The answer is accurate and useful .\n" \
               f"POOR: The answer is not helpful.\n" \
               f"NONE: The content contains no answer to the question.\n\n" \
               f"Answer with the response of COMPLETE, GOOD, POOR or NONE\n" \
               f"If the rating is GOOD or COMPLETE then also respond with an accurate,  helpful answer to the question, " \
               f"if the rating is POOR or NONE then no answer is required.\n" \
               f"Use the following template:\n" \
               f"RATING: Rating goes here EG GOOD\n" \
               f"REASON: Explain your reasoning here. Double check your answer.\n" \
               f"ANSWER: Helpful answer here for GOOD and COMPLETE answers only.  maximum 160 words."

    return template


def get_rating_question3(question):
    template = f"Carefully read the CONTENT provided. Use only the text from the CONTENT to write a helpful and " \
               f"accurate answer to the following question from an iMIS user. \n\n" \
               f"QUESTION: {question} \n\n" \
               f"Use the following rules to give a RATING to the answer: \n" \
               f"COMPLETE: There is a comprehensive answer to the question in the content.\n" \
               f"GOOD: The content contains a useful answer.\n" \
               f"POOR: The content does not contain a useful answer.\n" \
               f"NONE: The content contains no answer to the question.\n\n" \
               f"Answer with the response of COMPLETE, GOOD, POOR or NONE\n" \
               f"If the rating is GOOD or COMPLETE then also respond with an accurate,  helpful answer to the question, " \
               f"if the rating is POOR or NONE then no answer is required.\n" \
               f"Use the following template:\n" \
               f"RATING: Rating goes here EG GOOD\n" \
               f"REASON: Explain your reasoning here. Double check your answer.\n" \
               f"ANSWER: Helpful answer here for GOOD and COMPLETE answers only.  maximum 160 words."

    return template


def get_complete_question(question, words):
    template = f"Write a comprehensive and accurate answer to the question below using the knowledge provided from " \
               f"the iMIS EMS user documentation. \n " \
               f"Your answer must have a maximum of {words} words. \n " \
               f"QUESTION: '{question}' \n" \
               f"ANSWER:"
    return template


def get_reword_system():
    template = f"### BACKGROUND ### \n" \
               f"Your job is to write re-wordings of a question for the purpose of helping find an answer in a " \
               f"database.  Respond with 3 versions of the same question, using the template provided. Each version " \
               f"uses different words but has the exact same intent as the original, the same purpose and would " \
               f"require the same answer as the original question. The questions are from iMIS users. You don't need " \
               f"to mention iMIS in your versions of the question, it can be assumed all questions are already about " \
               f"iMIS. Write the question versions in as a single paragraph instead of numbering them. Make each " \
               f"question short and precise.\n" \
               f"### TEMPLATE ###\n" \
               f"ORIGINAL: Original question here.\n" \
               f"VERSIONS: Question version one. Question version two.  Question version three."

    return template


def get_three_questions_system():
    template = f"### INSTRUCTIONS ###\n" \
               f"Your job is to break a question up into its parts, in order to figure out if some content / text has " \
               f"the answer to the question.  You are answering the question ' To find the answer to your question in " \
               f"this text you might ask these three questions:'\n" \
               f"EXAMPLE\n" \
               f"QUESTION 'Do members have to pay their fees upfront?' \n" \
               f"RESPONSE: 'does the text mention members?  does the text discuss member fees?  Does the text talk " \
               f"about paying fees upfront?\n" \
               f"EXAMPLE\n" \
               f"QUESTION 'is there a limit to the number of images I can upload to a page?'\n" \
               f"RESPONSE: 'does the text talk about images?  does the text mention uploading images?  Does the text " \
               f"discuss limits on uploading?\n" \
               f"EXAMPLE\n" \
               f"QUESTION 'can I get a folder back if I delete it?'\n" \
               f"RESPONSE: 'Does the text mention deleting folders?  Does the text discuss the possibility of getting " \
               f"a deleted folder back?  Does the text provide any information on how to retrieve a deleted folder?'"

    return template


def get_three_questions(question):
    template = f"QUESTION: '{question}?' \n" \
               f"To find the answer to your question in this text you might ask these three short questions:\n" \
               f"RESPONSE:"

    return template


def get_reword_question(question):
    template = f"Write the original and 3 short differently worded versions of the question, they must all have the " \
               f"same intent, that all require the exact same answer. Do not number the versions. \n\n" \
               f"QUESTION:  '{question}?'\n" \
               f"Your answer should be in the following template:\n" \
               f"ORIGINAL: Original question here.\n" \
               f"VERSIONS: Question version one. Question version two.  Question version three."

    return template


def get_original_question(question):
    template = f"Return an improved version of the following question making sure to fix spelling and improve grammar." \
               f"QUESTION:  '{question}?' " \
               f"IMPROVED VERSION:"

    return template


def get_helpme_question(question):
    template = f"Help me find the answer to the question below, we are only allowed to use information from " \
               f"the documentation given If the answer isn't in the articles, thats ok, its better to say I " \
               f"can't find the answer then to give the wrong answer.  Make the answer {style_guide}. \n" \
               f"QUESTION: {question} \n" \
               f"ANSWER: (maximum {answer_words} words)"

    return template

def get_final_question(question):
    template = f"Use the content from the articles provided to write an answer the question " \
               f"below. The answer is aimed at an iMIS EMS user and must be {style_guide}.  Make the answer " \
               f"maximum {answer_words} words. If the question asks for a method or steps, use numbered steps when " \
               f"appropriate to illustrate your answer. If the answer is not in any of the articles respond with 'i don't know'." \
               f"QUESTION: {question} \n" \
               f"ANSWER:"

    return template


def get_question(question):
    # question = input("question:")

    if question == "quit":
        exit()
    elif len(question) > 6:
        answer = search_collection(question)

    st.write(answer)


def get_openai_response(system, question, temp=1, max_tokens=256):
    print("making openai call...")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"{system}"
                },
                {
                    "role": "user",
                    "content": f"{question}"
                }
            ],
            temperature=temp,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        response_text = response.choices[0].message.content

        return response_text

    except Exception as e:
        print("openai error: " + str(e))
        return "ERROR"

def run_prompt():
    style_guide = ""
    
    if answer_style:
        if answer_style == 'Friendly':
            style_guide = 'friendly, warm and polite'
        elif answer_style == 'Direct':
            style_guide = 'blunt, cold and direct'
        elif answer_style == 'Professional':
            style_guide = 'professional, respectful and comprehensive'
    
    if run_search or prompt != st.session_state['question']:
    
        st.session_state['question'] = prompt
        answer = ""
    
        if prompt == "quit":
            exit()
        elif len(prompt) > 6:
            print("search collection")
            answer = search_collection(prompt)
    
        #st.write(answer)

