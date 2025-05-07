import sys
import os
import json
import tqdm
import pymorphy3
import re
import torch
import ast

from bs4 import BeautifulSoup
from trl import SFTTrainer
from random import randrange
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

hf_api_key = os.getenv("HF_API_KEY")

login(hf_api_key)

morph = pymorphy3.MorphAnalyzer()

def lemmatize_text(text):
    text_without_punctuation = re.sub(r'[^\w\s]', '', text)

    words = text_without_punctuation.lower().split()

    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]

    return ' '.join(lemmatized_words)

def find_synonyms_feature_simple(name, concrete_complaint):
    if 'Синонимы' in concrete_complaint['Простой признак']:
        variants = [lemmatize_text(name)] + [lemmatize_text(variant) for variant in concrete_complaint['Простой признак']['Синонимы']['синоним']]
        return r"|".join(variants)
    else:
        return lemmatize_text(name)
    
def find_synonyms_feature_complex(name, concrete_complaint, pain_chest=False):
    if pain_chest:
        if 'Синонимы' in concrete_complaint['Составной признак']:
            variants = [lemmatize_text(name)] + [lemmatize_text("боли за грудиной")] + [lemmatize_text("на чувство наболевшего за грудиной")] + [lemmatize_text("боль за грудиной")] + [lemmatize_text(variant) for variant in concrete_complaint['Составной признак']['Синонимы']['синоним']]
            return r"|".join(variants)
        else:
            return lemmatize_text(name)
    else:
        if 'Синонимы' in concrete_complaint['Составной признак']:
            variants = [lemmatize_text(name)] + [lemmatize_text(variant) for variant in concrete_complaint['Составной признак']['Синонимы']['синоним']]
            return r"|".join(variants)
        else:
            return lemmatize_text(name)
        
def fill_result_feature_simple(pattern, complaint, concrete_complaint, result):

    if re.search(pattern, lemmatize_text(complaint['complaints'])):
        if {concrete_complaint[0]: {}} not in result["Output"]["Жалобы"]["Признак"]:
            result["Output"]["Жалобы"]["Признак"].append({concrete_complaint[0]: {}})

            if 'Качественные значения' in concrete_complaint[1]['Простой признак']:
                result["Output"]["Жалобы"]["Признак"][-1][concrete_complaint[0]]["Качественные значения"] = {"значение":["имеется"]}

    return result

def fill_result_feature_complex(pattern, complaint, concrete_complaint, result):
    if re.search(pattern, lemmatize_text(complaint['complaints'])):
        if {concrete_complaint[0]: {}} not in result["Output"]["Жалобы"]["Признак"]:
            result["Output"]["Жалобы"]["Признак"].append({concrete_complaint[0]: {}})
            result["Output"]["Жалобы"]["Признак"][-1][concrete_complaint[0]]['Характеристика'] = {}

            for characteristic in concrete_complaint[1]['Составной признак']['Характеристика'].items():
                value_list = []
                if (characteristic[1] != {}) and ("Качественные значения" in characteristic[1]["Тип возможных значений"]):

                    if "Значение" in characteristic[1]["Тип возможных значений"]["Качественные значения"]:
                        for v in characteristic[1]["Тип возможных значений"]["Качественные значения"]["Значение"].items():

                            if ('Синонимы' in v[1]) and (v[1] != {'Синонимы': {}}):
                                if "синоним (для значения)" in v[1]['Синонимы']:
                                    variants = [lemmatize_text(v[0])] + [lemmatize_text(variant) for variant in v[1]['Синонимы']['синоним (для значения)']]
                                    pattern =  r"|".join(variants)
                            else:
                                pattern = lemmatize_text(v[0])

                            if re.search(pattern, lemmatize_text(complaint['complaints'])):
                                value_list.append(v[0])

                    if len(value_list) != 0:
                        result["Output"]["Жалобы"]["Признак"][-1][concrete_complaint[0]]['Характеристика'][characteristic[0]] = {}
                        result["Output"]["Жалобы"]["Признак"][-1][concrete_complaint[0]]['Характеристика'][characteristic[0]]["Качественные значения"] = {"Значение": value_list}

            if result["Output"]["Жалобы"]["Признак"][-1][concrete_complaint[0]]['Характеристика'] == {}:
                result["Output"]["Жалобы"]["Признак"][-1][concrete_complaint[0]] = {}
    return result

def get_simple_json(complaint, complaints_groups):
    result = {"Output":{"Жалобы": {"Признак": []}}, "Input": complaint['complaints'], "file_name": complaint['file']}

    for complaints_group in complaints_groups.items():
        if 'Признак' in complaints_group[1]:
            for concrete_complaint in complaints_group[1]['Признак'].items():

                if 'Простой признак' in concrete_complaint[1]:
                    pattern = find_synonyms_feature_simple(concrete_complaint[0], concrete_complaint[1])
                    result = fill_result_feature_simple(pattern, complaint, concrete_complaint, result)

                elif 'Составной признак' in concrete_complaint[1]:

                    if concrete_complaint[0] == "Артериальное давление":
                        if re.search("АД", complaint['complaints']):
                            if concrete_complaint[0] not in result["Output"]["Жалобы"]["Признак"]:
                                result["Output"]["Жалобы"]["Признак"].append({concrete_complaint[0]: {}})

                    else:
                        pattern = find_synonyms_feature_complex(concrete_complaint[0], concrete_complaint[1])
                        result = fill_result_feature_complex(pattern, complaint, concrete_complaint, result)

                else:
                    pattern = lemmatize_text(concrete_complaint[0])

                    if re.search(pattern, lemmatize_text(complaint['complaints'])):
                        if {concrete_complaint[0]: {}} not in result["Output"]["Жалобы"]["Признак"]:
                            result["Output"]["Жалобы"]["Признак"].append({concrete_complaint[0]: {}})

        if 'Группа признаков' in complaints_group[1]:
            for feature_group in complaints_group[1]['Группа признаков'].items():
                for concrete_complaint in feature_group[1]['Признак'].items():

                    if 'Простой признак' in concrete_complaint[1]:
                        pattern = find_synonyms_feature_simple(concrete_complaint[0], concrete_complaint[1])
                        result = fill_result_feature_simple(pattern, complaint, concrete_complaint, result)

                    elif 'Составной признак' in concrete_complaint[1]:
                        if concrete_complaint[0] == "Артериальное давление":
                            if re.search("АД", complaint['complaints']):
                                if concrete_complaint[0] not in result["Output"]["Жалобы"]["Признак"]:
                                    result["Output"]["Жалобы"]["Признак"].append({concrete_complaint[0]: {}})

                        elif concrete_complaint[0] == "Боль в грудной клетке":
                            pattern = find_synonyms_feature_complex(concrete_complaint[0], concrete_complaint[1], pain_chest = True)
                            result = fill_result_feature_complex(pattern, complaint, concrete_complaint, result)

                        else:
                            pattern = find_synonyms_feature_complex(concrete_complaint[0], concrete_complaint[1])
                            result = fill_result_feature_complex(pattern, complaint, concrete_complaint, result)

                    else:
                        pattern = lemmatize_text(concrete_complaint[0])

                        if re.search(pattern, lemmatize_text(complaint['complaints'])):
                            if {concrete_complaint[0]: {}} not in result["Output"]["Жалобы"]["Признак"]:
                                result["Output"]["Жалобы"]["Признак"].append({concrete_complaint[0]: {}})

    if len(result["Output"]["Жалобы"]["Признак"]) != 0:
        return result
    else:
        return None

def extract_complaints(html_file):
    with open(html_file, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
    
    center_tags = soup.find_all("center")
    for center in center_tags:
        if "Жалобы:" in center.get_text():
            complaints_text = center.find_next_sibling(string=True)
            if complaints_text:
                return complaints_text.strip()
    return "Жалобы не найдены"

def process_files(directory):
    complaints_list = []
    for filename in tqdm.tqdm(os.listdir(directory)):
        if filename.endswith(".html"):
            filepath = os.path.join(directory, filename)
            complaints = extract_complaints(filepath)
            complaints_list.append({"file": filename, "complaints": complaints})
            print(complaints)
    return complaints_list

def remove_duplicate_keys_by_name(symptom_list):
    result = {}
    for item in symptom_list:
        for key, value in item.items():
            if key not in result or (result[key] == {} and value != {}):
                result[key] = value
    return [{k: v} for k, v in result.items()]

def extract_response_dict(text: str) -> dict:
    match = re.search(r'### Response:\s*(\{.*)', text, re.DOTALL)
    if not match:
        raise ValueError("Блок ### Response: не найден или не содержит словарь.")

    response_text = match.group(1).strip()

    bracket_count = 0
    end_index = None
    for i, char in enumerate(response_text):
        if char == '{':
            bracket_count += 1
        elif char == '}':
            bracket_count -= 1
            if bracket_count == 0:
                end_index = i + 1
                break

    if end_index is None:
        return {"Жалобы": {}}
        raise ValueError("Не удалось найти полный словарь после ### Response.")

    dict_str = response_text[:end_index]

    try:
        result = ast.literal_eval(dict_str)
        if not isinstance(result, dict):
            raise ValueError("Результат не является словарем.")
        return result
    except Exception as e:
        raise ValueError(f"Ошибка при разборе словаря: {e}")

def get_simple_json_llm(complaint):

    finetuned_model = AutoPeftModelForCausalLM.from_pretrained(
        "mistral_instruct_qa_v5",
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained("mistral_instruct_qa_v5")

    text_1 = """### Instruction:
        Извлеки жалобы из текста и приведи в структурированный вид.
        Например текст: смешанная одышка при минимальной физической нагрузке- изменение положения туловища, ходьба до 3-5 метров, одышка уменьшается после ингаляции сальбутамола. Периодический кашель , больше по утрам, с отделением скудной вязкой слизистой или слизистогнойной мокроты. Слабость, утомляемость.
        структурированный вид: {'Жалобы': {'Признак': [{'Утомляемость': {'Качественные значения': {'значение': ['имеется']}}},
        {'Общая слабость': {'Характеристика': {'Периодичность': {'Качественные значения': {'Значение': ['периодически']}}}}},
        {'Кашель': {'Характеристика': {'Время появления': {'Качественные значения': {'Значение': ['периодический',
            'утро']}}}}},
        {'Мокрота': {'Характеристика': {'Характер': {'Качественные значения': {'Значение': ['слизистая',
            'гнойная',
            'слизисто-гнойная']}}}}},
        {'Одышка': {'Характеристика': {'Характер': {'Качественные значения': {'Значение': ['затруднение вдоха и выдоха']}}}}}]}}
        """
    text_2 =  f"""
        ### Input
        {complaint['complaints']}

        ### Response:
    """

    prompt = text_1 + text_2

    encoded_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    model_inputs = encoded_input.to('cuda')

    generated_ids = finetuned_model.generate(**model_inputs, max_new_tokens=256, do_sample=True, pad_token_id=tokenizer.eos_token_id)

    decoded_output = tokenizer.batch_decode(generated_ids)

    res_output = extract_response_dict(decoded_output[0])

    print("\n Результат работы LLM:")
    print(f"Запрос: {complaint['complaints']}")
    print(f"Ответ: {res_output}")
    print("="*100)

    result = {"Output": res_output, "Input": complaint['complaints'], "file_name": complaint['file']}

    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python parser.py <папка_с_html>")
    else:
        directory = sys.argv[1]
        json_file = "complaints.json"
        complaints = process_files(directory)
        
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(complaints, f, ensure_ascii=False, indent=4)
        
        with open(json_file, "r", encoding="utf-8") as f:
            complaints = json.load(f)

        with open("База медицинской терминологии и наблюдений 2020 - Практика 2025.simple.json", "r", encoding="utf-8") as f:
            base_term = json.load(f)   

        dataset = []
        complaints_groups = base_term['Жалобы']['Группа признаков'] 
        
        for complaint in tqdm.tqdm(complaints):

            result = get_simple_json_llm(complaint)

            if result:
                dataset.append(result)
            else:
                result = get_simple_json(complaint, complaints_groups)

                if result:
                    dataset.append(result)
        
        for i in range(0, len(dataset)):
            symptoms = dataset[i]['Output']['Жалобы']['Признак']
            dataset[i]['Output']['Жалобы']['Признак'] = remove_duplicate_keys_by_name(symptoms)

        with open("simple.json", "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(f"Извлеченный текст жалоб сохранен в {json_file}")
        print(f"Упрощенный json сохранен в simple.json")