from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

### ML MODEL START


# ! pip install transformers
# ! pip install datasets

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import os
import clang
from clang.cindex import *
from copy import deepcopy

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from datasets import Dataset

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

Config.set_library_file("/home/dipu/anaconda3/lib/python3.9/site-packages/clang/native/libclang.so")

id2label = {0: "CORRECT", 1: "BUGGY"}
label2id = {"CORRECT": 0, "BUGGY": 1}


tokenizer = AutoTokenizer.from_pretrained('dipudl/codet5-base')

model = AutoModelForSequenceClassification.from_pretrained("codet5_distilbert-base_148k-files_10ep",
                                                           local_files_only=True,
                                                           num_labels=2,
                                                           id2label=id2label,
                                                           label2id=label2id)

def tokenize_text(examples):
    return tokenizer(examples["full_text"], truncation=True, max_length=100, padding=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}


trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

def get_function_params(root, function_name, result):
    for node in root.walk_preorder():
        try:
            if node.kind == CursorKind.FUNCTION_DECL and node.spelling == function_name:
                for c in node.get_children():
                    if c.kind == CursorKind.PARM_DECL:
                        result.append({"name": c.spelling, "data_type": c.type.spelling})
                return
        except ValueError as e:
            pass
            # print("Error:", e)


def get_called_functions(root, result):
    for node in root.walk_preorder():
        try:
            if node.kind == CursorKind.CALL_EXPR:
                # "location": node.extent
                current_function = {"name": node.spelling, "location": node.extent, "return_type": node.type.spelling, "args": []}

                for c in node.get_arguments():
                    current_arg = list(c.get_tokens())[0].spelling if len(list(c.get_tokens())) > 0 else c.spelling

                    if len(current_arg) >= 3 and current_arg.startswith('\"') and current_arg.endswith('\"'):
                        current_arg = '\"' + current_arg[1:-1].replace('\"', "\"\"") + '\"'

                    current_function["args"].append({"name": current_arg, "data_type": c.type.spelling, "cursor_kind": c.kind})
                    # current_function["args"].append({"name": c.spelling, "data_type": c.type.spelling, "cursor_kind": c.kind})
                    # print(node.location)

                current_param_list = []
                if len(current_function["args"]) == 2 and (current_function["args"][0]["data_type"] == current_function["args"][1]["data_type"]):
                    get_function_params(root, node.spelling, current_param_list)
                current_function["params"] = current_param_list

                result.append(current_function)

        except ValueError:
            pass


def get_swap_possible_functions(code):
    with open('evaluation.c', 'w') as f:
        f.write(code)
    
    index = clang.cindex.Index.create()
    root_cursor = index.parse('evaluation.c').cursor
    
    function_list = []
    get_called_functions(root_cursor, function_list)
    function_list
    
    full_text, function_name, start_line, start_column, end_line, end_column = [], [], [], [], [], []
    for function in function_list:
        
        if len(function["args"]) == 2 and (function["args"][0]["data_type"] == function["args"][1]["data_type"]) and (function["args"][0]["name"] != function["args"][1]["name"]):
            
            filtered_data = [function["name"], function["args"][0]["name"], function["args"][1]["name"],
                              function["args"][0]["data_type"]]
            
            if(len(function["params"]) == 2):
                filtered_data.append(function["params"][0]["name"])
                filtered_data.append(function["params"][1]["name"])
            else:
                filtered_data.append("[UNK]")
                filtered_data.append("[UNK]")
            
            filtered_data = " ".join(filtered_data)
            
            full_text.append(filtered_data)
            function_name.append(function["name"])
            
            loc = function["location"]
            start_line.append(loc.start.line)
            start_column.append(loc.start.column)
            end_line.append(loc.end.line)
            end_column.append(loc.end.column)
    
    return full_text, function_name, start_line, start_column, end_line, end_column


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def generate_prediction(code):
    full_text, function_name, start_line, start_column, end_line, end_column = get_swap_possible_functions(code)

    evaluation_df = pd.DataFrame({"full_text": full_text, "function_name": function_name,
                                "start_line": start_line,
                "start_column": start_column, "end_line": end_line,
                "end_column": end_column})

    evaluation_dataset = Dataset.from_pandas(evaluation_df)

    evaluation_dataset = evaluation_dataset.map(tokenize_text, remove_columns=["full_text"])

    prediction = trainer.predict(evaluation_dataset)

    output = []
    for i in range(len(prediction.predictions)):
        sm = softmax(prediction.predictions[i])
        bug = np.argmax(sm)
        output.append({"function_name": evaluation_dataset[i]["function_name"],
                    "start_line": evaluation_dataset[i]["start_line"],
                    "start_column": evaluation_dataset[i]["start_column"],
                    "end_line": evaluation_dataset[i]["end_line"],
                    "end_column": evaluation_dataset[i]["end_column"],
                    "is_buggy": bug,
                    "probability": sm[bug]
                    }
                    )
    return output


print(generate_prediction("""
#include <stdio.h>


void getSum(int n1, int n2)
{
	int sum = n1 + n2;
	return sum;
}

int main()
{
	int a = 5, b = 7, x = 6;
	int result = 555;
    getSum(a, 7);
    char result[] = "test\0";
    int test = justDoThis(a, b);
    okayGood(x, 500);
	printf("Sum is: %d", result);
	return 0;
}
"""))


#### ML MODEL END

class Item(BaseModel):
    code: str = None

@app.post("/analyze")
async def analyze(code: Item):
    print(code.code)
    print(tokenizer.tokenize("int a = 0;"))
    prediction = generate_prediction(code.code)
    # output={"hello": "hi"}
    output = {"analysis": prediction}
    print(output)
    # output = {"analysis": [{'function_name': 'getSum', 'start_line': 15, 'start_column': 5, 'end_line': 15, 'end_column': 17, 'is_buggy': 0, 'probability': 0.99971753}, {'function_name': 'justDoThis', 'start_line': 17, 'start_column': 16, 'end_line': 17, 'end_column': 32, 'is_buggy': 1, 'probability': 0.7907314}, {'function_name': 'okayGood', 'start_line': 18, 'start_column': 5, 'end_line': 18, 'end_column': 21, 'is_buggy': 1, 'probability': 0.70102}]}
    return JSONResponse(content=str(output))