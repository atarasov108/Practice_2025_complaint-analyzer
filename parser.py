import sys
import os
import json
import tqdm
import pymorphy3
import re
import torch
import ast
import random

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

try:
    finetuned_model_mistalv6 = AutoPeftModelForCausalLM.from_pretrained(
        "mistral_instruct_qa_v6",
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer_mistalv6 = AutoTokenizer.from_pretrained("mistral_instruct_qa_v6")
except:
    finetuned_model_mistalv6 = AutoPeftModelForCausalLM.from_pretrained(
        "/content/complaint-analyzer/mistral_instruct_qa_v6",
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer_mistalv6 = AutoTokenizer.from_pretrained("/content/complaint-analyzer/mistral_instruct_qa_v6")
else:
    print("Не удалось загрузить модель mistral")

try:
    finetuned_model_openchatv1 = AutoPeftModelForCausalLM.from_pretrained(
        "openchat_v1",
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer_openchatv1 = AutoTokenizer.from_pretrained("openchat_v1")
except:
    finetuned_model_openchatv1 = AutoPeftModelForCausalLM.from_pretrained(
        "/content/complaint-analyzer/openchat_v1",
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer_openchatv1 = AutoTokenizer.from_pretrained("/content/complaint-analyzer/openchat_v1")
else:
    print("Не удалось загрузить модель openchat")

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

    dict_str = response_text[:end_index]

    try:
        result = ast.literal_eval(dict_str)
        if not isinstance(result, dict):
            print("Результат не является словарем.")
            return {"Жалобы": {}}
        return result
    except Exception as e:
        print(f"Ошибка при разборе словаря: {e}")
        return {"Жалобы": {}}

def get_simple_json_llm(complaint):

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

    no_complaints_list = [
        "Жалобы не найдены", "на момент осмотра нет.", "на момент осмотра нет", 
        "нет", "Нет", "нет.", 
        "Нет.", "на момент осмотра активно не предъявляет", "активно нет", 
        "активно не предъявляет", "Активно жалоб не предъявляет из-за тяжести состояния", "",
        "Активных жалоб на момент осомтра не предъявляет.", "активно не предъявляет.", ".", " "
    ]
    prompt = text_1 + text_2
    if complaint['complaints'] in no_complaints_list:
        return {"Output": {'Жалобы': {'Признак': []}}, "Input": complaint['complaints'], "file_name": complaint['file']}
    try:
        encoded_input = tokenizer_mistalv6(prompt, return_tensors="pt", add_special_tokens=True)
        model_inputs = encoded_input.to('cuda')

        generated_ids = finetuned_model_mistalv6.generate(**model_inputs, max_new_tokens=256, do_sample=True, pad_token_id=tokenizer_mistalv6.eos_token_id)

        decoded_output = tokenizer_mistalv6.batch_decode(generated_ids)

        res_output = extract_response_dict(decoded_output[0])

        print("\n Результат работы LLM Mistral:")
        print(f"Запрос: {complaint['complaints']}")
        print(f"Ответ: {res_output}")
        print("\n")

        encoded_input = tokenizer_openchatv1(prompt, return_tensors="pt", add_special_tokens=True)
        model_inputs = encoded_input.to('cuda')

        generated_ids = finetuned_model_openchatv1.generate(**model_inputs, max_new_tokens=256, do_sample=True, pad_token_id=tokenizer_openchatv1.eos_token_id)

        decoded_output = tokenizer_openchatv1.batch_decode(generated_ids)

        res_output_openchatv1 = extract_response_dict(decoded_output[0])

        print("\n Результат работы LLM OpenChat:")
        print(f"Запрос: {complaint['complaints']}")
        print(f"Ответ: {res_output_openchatv1}")
        print("="*100)

        
        
        result = {"Output": res_output, "Input": complaint['complaints'], "file_name": complaint['file']}

        return result
    except:
        return None

def find_key_with_path(data, target_key):
    results = []

    def _search(obj, path):
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = path + [key]
                if key == target_key:
                    results.append((current_path))
                _search(value, current_path)
        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                _search(item, path + [index])

    _search(data, [])
    res_path = None
    for path in results:
        res_path = "/".join(map(str, path))
    if res_path:
        return res_path+";"
    else:
        return None

def fill_complex_complaint(universal_json, name, original_complaint, value):
    universal_json["successors"].append(
        {
            "id" : random.randint(10**14, 10**15 - 1),
            "name" : name,
            "type" : "НЕТЕРМИНАЛ",
            "meta" : "Признак",
            "original" : original_complaint,
            "successors" :
            []
        }
    )
    for charact in value['Характеристика']:
        original_value = original_complaint.replace(";", f"/Составной признак/{charact};")
        universal_json["successors"][-1]["successors"].append(
            {
                "id" : random.randint(10**14, 10**15 - 1),
                "name" : charact,
                "type" : "НЕТЕРМИНАЛ",
                "meta" : "Характеристика",
                "original" : original_value,
                "successors" :
                [            
                    {
                        "id" : random.randint(10**14, 10**15 - 1),
                        "name" : "Качественные значения",
                        "type" : "НЕТЕРМИНАЛ",
                        "meta" : "Качественные значения",
                        "successors" :
                        []
                    }
                ]
            }
        )

        for qual_value in value['Характеристика'][charact]["Качественные значения"]["Значение"]:
            original_qual_value = original_value.replace(";", f"/Тип возможных значений/Качественные значения/{qual_value};")
            universal_json["successors"][-1]["successors"][-1]["successors"][-1]["successors"].append(
                {
                    "id" : random.randint(10**14, 10**15 - 1),
                    "value" : qual_value,
                    "type" : "ТЕРМИНАЛ-ЗНАЧЕНИЕ",
                    "valtype" : "STRING",
                    "meta" : "значение",
                    "original" : original_qual_value
                } 
            )




def fill_simple_complaint(universal_json, name, original_complaint, value):
    original_value = find_key_with_path(origin_search, name)
    original_value = original_value.replace("База медицинской терминологии и наблюдений 2020 - Практика 2025", "База медицинской терминологии и наблюдений 2020 - Практика 2025$")
    original_value = original_value.replace(";", "/Простой признак/Качественные значения/имеется;")
    original_value = original_value.replace("/Группа наблюдений/", "/")
    original_value = original_value.replace("/Группа признаков/", "/")
    original_value = original_value.replace("/Признак/", "/")
    universal_json["successors"].append(
        {
            "id" : random.randint(10**14, 10**15 - 1),
            "name" : name,
            "type" : "НЕТЕРМИНАЛ",
            "meta" : "Признак",
            "original" : original_complaint,
            "successors" :
            [
                {
                    "id" : random.randint(10**14, 10**15 - 1),
                    "name" : "Качественные значения",
                    "type" : "НЕТЕРМИНАЛ",
                    "meta" : "Качественные значения",
                    "successors" :
                    [
                        {
                            "id" : random.randint(10**14, 10**15 - 1),
                            "value" : "имеется",
                            "type" : "ТЕРМИНАЛ-ЗНАЧЕНИЕ",
                            "valtype" : "STRING",
                            "meta" : "значение",
                            "original" : original_value
                        }
                    ]
                }
             ]
        }
    )

def get_universal_json(simple_json):

    random.randint(10**14, 10**15 - 1)

    universal_json = {
        "id" : random.randint(10**14, 10**15 - 1),
        "name" : "Жалобы",
        "type" : "НЕТЕРМИНАЛ",
        "meta" : "Жалобы",
        "successors" :
        []
    }

    if "Признак" not in simple_json['Жалобы']:
        return universal_json

    for complaint in simple_json['Жалобы']['Признак']:

        for name, value in complaint.items():

            original_complaint = find_key_with_path(origin_search, name)
            if original_complaint:
                original_complaint = original_complaint\
                    .replace("База медицинской терминологии и наблюдений 2020 - Практика 2025", "База медицинской терминологии и наблюдений 2020 - Практика 2025$")
                original_complaint = original_complaint.replace("/Группа наблюдений/", "/")
                original_complaint = original_complaint.replace("/Группа признаков/", "/")
                original_complaint = original_complaint.replace("/Признак/", "/")

                if 'Характеристика' in value:
                    fill_complex_complaint(universal_json, name, original_complaint, value)
                elif 'Качественные значения' in value:
                    fill_simple_complaint(universal_json, name, original_complaint, value)
                else:
                    universal_json["successors"].append(
                        {
                            "id" : random.randint(10**14, 10**15 - 1),
                            "name" : name,
                            "type" : "НЕТЕРМИНАЛ",
                            "meta" : "Признак",
                            "original" : original_complaint,
                            "successors" :
                            []
                        }
                    )
            else:
                continue

    return universal_json

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

        try:
            with open("База медицинской терминологии и наблюдений 2020 - Практика 2025.simple.json", "r", encoding="utf-8") as f:
                base_term = json.load(f)
        except:
            with open("/content/complaint-analyzer/База медицинской терминологии и наблюдений 2020 - Практика 2025.simple.json", "r", encoding="utf-8") as f:
                base_term = json.load(f)

        dataset = []
        complaints_groups = base_term['Жалобы']['Группа признаков'] 
        
        for complaint in tqdm.tqdm(complaints):

            result = get_simple_json_llm(complaint)

            if result:
                print("Json успешно сгенерирован")
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

        universal_dataset = []
        universal_dataset_platform = []
        try:
            with open("origin_search.simple.json", "r", encoding="utf-8") as f:
                origin_search = json.load(f)   
        except:
            with open("/content/complaint-analyzer/origin_search.simple.json", "r", encoding="utf-8") as f:
                origin_search = json.load(f)  
        
        for simple_dict in dataset:
            simple_dict['Output'] = get_universal_json(simple_dict['Output'])
            universal_dataset.append(simple_dict)
            universal_dataset_platform.append({
                        "id" : random.randint(10**14, 10**15 - 1),
                        "name" : f"История болезни 2024-11-14 03:02:58:{random.randint(10**5, 10**6 - 1)}",
                        "type" : "НЕТЕРМИНАЛ",
                        "meta" : "История болезни или наблюдений v.4",
                        "successors" :
                        [simple_dict['Output']]}
                    )

        universal_dict = {
            "title" : "Архив ИБ - Практика 2025 - Тест1",
            "code" : "4640953873793518062",
            "path" : "lemesh.ve@dvfu.ru / Мой Фонд / Архив ИБ - Практика 2025 - Тест1$;",
            "date" : "04.05.2025-19:55:56.734",
            "creation" : "04.03.2025-19:26:46.534",
            "owner_id" : 41,
            "json_type" : "universal",
            "ontology" : "Онтология электронной медицинской карты V.4 - Практика 2025$;",
            "id" : 603039178162180,
            "name" : "Архив ИБ - Практика 2025 - Тест1",
            "type" : "КОРЕНЬ",
            "meta" : "Онтология электронной медицинской карты V.4 - Практика 2025",
            "successors" :
            [{
                "id" : 603039178162184,
                "name" : "Врачебные осмотры, консультации, истории болезни",
                "type" : "НЕТЕРМИНАЛ",
                "meta" : "Врачебные осмотры, консультации, истории болезни",
                "successors" :
                universal_dataset_platform}
            ]
        }
        with open("universal.json", "w", encoding="utf-8") as f:
            json.dump(universal_dict, f, ensure_ascii=False, indent=2)

        print(f"Извлеченный текст жалоб сохранен в {json_file}")
        print(f"Упрощенный json сохранен в simple.json")
        print(f"Универсальный json сохранен в universal.json")