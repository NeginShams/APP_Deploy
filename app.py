import json
import torch
import re
from parsivar import Normalizer
from bs4 import BeautifulSoup
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import pipeline
from haystack.nodes import FARMReader
from haystack.pipelines import Pipeline
from haystack import Pipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import PreProcessor
from haystack.nodes import DensePassageRetriever, BM25Retriever
from haystack.nodes import JoinDocuments, SentenceTransformersRanker
from flask import Flask, render_template
from flask import request


app = Flask(__name__)

preprocessor = PreProcessor(
    split_by="word",
    language = 'fa',
    # split_length=200,
    # split_overlap=0,
    split_respect_sentence_boundary=False,
    clean_empty_lines=False,
    clean_whitespace=False,
)

document_store = FAISSDocumentStore.load(index_path="/home/negin/Quran_Index/Quran_Index/index.faiss")

sparse_retriever = BM25Retriever(document_store=document_store)
dense_retriever = DensePassageRetriever(document_store=document_store,
                                query_embedding_model="/home/negin/dpr_parsbert/query_encoder",
                                passage_embedding_model="/home/negin/dpr_parsbert/passage_encoder",
                                use_gpu=True,
                                max_seq_len_passage=256,
                                embed_title=True)


join_documents = JoinDocuments(join_mode="concatenate")
rerank = SentenceTransformersRanker(model_name_or_path="NeginShams/cross_encoder_v2")


def Question_classifier(question):
  if 'سوره' in question and 'آیه' in question:
    return 'referential'
  else:
    return 'non_referential'

def find_verse_address(question):

    with open('/home/negin/QuranInfo.json', 'r', encoding = 'utf_8_sig') as json_file:
        quran_data_list = json.load(json_file)

    quran_info = {}

    for each in quran_data_list:
        quran_info[each['chapter_name']] = each['chapter_number']

    spl_word = 'آیه'
    res = question[question.find(spl_word)+len(spl_word):]
    verse_number = res.split()[0]

    spl_word = 'سوره'
    res = question[question.find(spl_word)+len(spl_word):]
    chapter_name = res.split()[0]
    chapter_name = chapter_name.replace("،","")

    if chapter_name == '‌ی':
        chapter_name = res.split()[1]

    if chapter_name == 'صاد':
        chapter_name = 'ص'

    if chapter_name == 'قاف':
        chapter_name = 'ق'

    if chapter_name == 'آل‌عمران':
        chapter_name = 'آل عمران'

    if chapter_name == 'انبیا':
        chapter_name = "انبیاء"

    if chapter_name == 'شرح':
        chapter_name = "انشراح"


    chapter_name = chapter_name.replace("،","")

    # verse_number = int(verse_number)
    try:
        chapter_number = quran_info[chapter_name]
    except:
        my_normalizer = Normalizer()
        chapter_name = my_normalizer.normalize(chapter_name)
        chapter_number = quran_info[chapter_name]

    if verse_number == 'اول':
        verse_number = '1'

    if verse_number == 'دوم':
        verse_number = '2'

    if verse_number == 'سوم':
        verse_number = '3'

    if verse_number == 'هفتم':
        verse_number = '7'

    if verse_number == 'نهم':
        verse_number = '9'

    verse_number = verse_number.replace("سوره","")

    verse_number = str(int(verse_number))

    verse_id = 's' + chapter_number + '.' + verse_number
    return verse_id


def answer_ensemble(context, question):
  checkpoint_list = ['NeginShams/albert-Quran_QA',
                  'NeginShams/xlm-roberta-Quran_QA',
                    'NeginShams/mbert-Quran_QA',
                    'NeginShams/parsbert-Quran_QA']
  results = []
  device = 0 if torch.cuda.is_available() else -1  # -1 for CPU

  for checkpoint in checkpoint_list:
    question_answerer = pipeline("question-answering", model=checkpoint, device=device)
    result = question_answerer(question=question, context=context)
    result['model_name'] = checkpoint
    results.append(result)
  # print(results)

  for i in range(len(results)):
      maximum = 0
      answer = ''
      for result in results:
        if result['score'] > maximum:
          maximum = result['score']
          answer = result['answer']
  # print(maximum)
  return answer


def qa_pipeline(question):

  checkpoint_list = ['NeginShams/albert-Quran_QA','NeginShams/xlm-roberta-Quran_QA', 'NeginShams/mbert-Quran_QA','NeginShams/parsbert-Quran_QA']
  # checkpoint_list = ['NeginShams/xlm-roberta-Quran_QA',]
  results = []
  for checkpoint in checkpoint_list:
    reader = FARMReader(model_name_or_path=checkpoint, top_k=4, use_gpu=True)
    pipeline = Pipeline()
    pipeline.add_node(component=sparse_retriever, name="SparseRetriever", inputs=["Query"])
    pipeline.add_node(component=dense_retriever, name="DenseRetriever", inputs=["Query"])
    pipeline.add_node(component=join_documents, name="JoinDocuments", inputs=["SparseRetriever", "DenseRetriever"])
    pipeline.add_node(component=rerank, name="ReRanker", inputs=["JoinDocuments"])
    pipeline.add_node(component=reader, name="Reader", inputs=["ReRanker"])

    k=1
    prediction = pipeline.run(
        query=question, params={"SparseRetriever": {"top_k": k},
                                "DenseRetriever": {"top_k": k},
                                "JoinDocuments": {"top_k_join": 2*k},
                                "ReRanker": {"top_k": k},
                                "Reader": {"top_k": 1}}
    )

    result = {}
    answer_dict = prediction['answers'][0].__dict__
    result['answer'] = answer_dict['answer']
    result['score'] = answer_dict['score']
    result['model_name'] = checkpoint
    results.append(result)

  # print(results)

  for i in range(len(results)):
      maximum = 0
      answer = ''
      for result in results:
        if result['score'] > maximum:
          maximum = result['score']
          answer = result['answer']
  # print(maximum)
  return answer


def find_answer(question):
  if Question_classifier(question) == 'referential':
    verse_id = find_verse_address(question)
    file_path = '/home/negin/makarem.xml'
    with open(file_path, 'r', encoding="utf8") as f:
      content = f.read()
    soup= BeautifulSoup(content, 'xml')
    # print(verse_id)
    verse_text = soup.find(id=verse_id).contents[0]
    context = re.sub("[\(\[].*?[\)\]]", "", verse_text)
    # print(context)

    final_answer = answer_ensemble(context, question)

    print('@@@@@@@@@@@@@@@@')
    print(final_answer)
    print('@@@@@@@@@@@@@@@@')

  else:
    final_answer = qa_pipeline(question)

    print('*************************')
    print('\n'+ final_answer)
    print('*************************')

  return final_answer


@app.route('/hello')
def hello_world():
   return render_template('page.html')

@app.route('/question_handle', methods = ['POST'])
def question_handle():
  if request.method == 'POST':
    question = request.form.get('name')
    ans = find_answer(question)
    return ans



# if __name__ == '__main__':
   

# preprocessor = PreProcessor(
#     split_by="word",
#     language = 'fa',
#     # split_length=200,
#     # split_overlap=0,
#     split_respect_sentence_boundary=False,
#     clean_empty_lines=False,
#     clean_whitespace=False,
# )

# document_store = FAISSDocumentStore.load(index_path="/home/negin/Quran_Index/Quran_Index/index.faiss")

# sparse_retriever = BM25Retriever(document_store=document_store)
# dense_retriever = DensePassageRetriever(document_store=document_store,
#                                 query_embedding_model="/home/negin/dpr_parsbert/query_encoder",
#                                 passage_embedding_model="/home/negin/dpr_parsbert/passage_encoder",
#                                 use_gpu=True,
#                                 max_seq_len_passage=256,
#                                 embed_title=True)


# join_documents = JoinDocuments(join_mode="concatenate")
# rerank = SentenceTransformersRanker(model_name_or_path="NeginShams/cross_encoder_v2")


# ans = find_answer("آیه 45 سوره‌ی مدثر به خطر همرنگی و همنشینی با چه کسانی اشاره دارد؟")

# print('answer:' + str(ans))

print("hello world")

app.run(port=5000)