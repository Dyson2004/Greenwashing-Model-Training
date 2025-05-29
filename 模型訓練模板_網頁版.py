import streamlit as st
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers.trainer_callback import TrainerCallback
from datasets import load_dataset, Features, Value, ClassLabel
import seaborn as sns
import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import os
print("Current working directory:", os.getcwd())
from matplotlib import rcParams
import matplotlib.pyplot as plt

# 設置中文字體
rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 可選的預訓練模型（僅用於訓練）
MODEL_OPTIONS = {
    "ClimateBERT": "climatebert/distilroberta-base-climate-f",
    "BERT Base": "bert-base-uncased"
}

# 標籤對應表（四分類）
LABELS = ["低風險漂綠", "中風險漂綠", "高風險漂綠", "極高風險漂綠"]

# 初始化 session_state 儲存模型和 tokenizer
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'trained_tokenizer' not in st.session_state:
    st.session_state.trained_tokenizer = None
if 'model_path' not in st.session_state:
    st.session_state.model_path = None

# 載入模型和 tokenizer
def load_model(model_source, model_name=None, model_path=None):
    if model_source == "預設模型":
        if model_name not in MODEL_OPTIONS:
            raise ValueError(f"無效的預設模型名稱：{model_name}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_OPTIONS[model_name])
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_OPTIONS[model_name], num_labels=4)
    else:  # 自定義模型路徑
        if not os.path.exists(model_path):
            raise ValueError(f"指定的模型路徑 {model_path} 不存在！")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=4)
    return tokenizer, model

# 學習曲線回調類
class LearningCurveCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
        self.epochs = []

    def on_epoch_end(self, args, state, control, **kwargs):
        if not state.log_history:
            return control
        log = state.log_history[-1]
        train_loss = log.get('loss', None)
        eval_loss = log.get('eval_loss', None)
        learning_rate = log.get('learning_rate', None)
        if train_loss is not None:
            self.train_losses.append(train_loss)
        else:
            self.train_losses.append(0)
        if eval_loss is not None:
            self.eval_losses.append(eval_loss)
        else:
            self.eval_losses.append(0)
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        else:
            self.learning_rates.append(0)
        self.epochs.append(len(self.epochs) + 1)
        return control

    def plot_learning_curves(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.epochs, self.train_losses, label='訓練損失', marker='o')
        plt.plot(self.epochs, self.eval_losses, label='驗證損失', marker='o')
        plt.title('模型損失曲線')
        plt.xlabel('訓練週期')
        plt.ylabel('損失')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.epochs, self.learning_rates, label='學習率', marker='o', color='green')
        plt.title('學習率變化')
        plt.xlabel('訓練週期')
        plt.ylabel('學習率')
        plt.legend()
        plt.tight_layout()
        curve_path = "learning_curves.png"
        plt.savefig(curve_path)
        plt.close()
        return curve_path

# 評估模型並生成混淆矩陣
def evaluate_model(model, tokenizer, data_path):
    dataset = load_dataset("json", data_files=data_path)
    all_data = list(dataset["train"])
    _, val_data = train_test_split(all_data, test_size=0.2, random_state=42)
    
    def preprocess_function(examples):
        result = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
        result["labels"] = examples["label"]
        return result
    
    tokenized_val = list(map(preprocess_function, val_data))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    true_labels = []
    predictions = []
    
    for example in tokenized_val:
        inputs = {
            "input_ids": torch.tensor(example["input_ids"]).unsqueeze(0).to(device),
            "attention_mask": torch.tensor(example["attention_mask"]).unsqueeze(0).to(device)
        }
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()
        true_labels.append(example["labels"])
        predictions.append(prediction)
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(true_labels, predictions, labels=[0, 1, 2, 3])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
    plt.title('混淆矩陣')
    plt.xlabel('預測標籤')
    plt.ylabel('真實標籤')
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    
    result = f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-score: {f1:.4f}"
    return result, cm_path

# 更新 train_model，接受動態參數
def train_model(model_source, model_name, model_path, data_file, train_batch_size, eval_batch_size, num_epochs):
    if data_file is None:
        return "錯誤：請上傳數據檔案！", None, None, None, None, None

# 驗證 JSON 檔案
    try:
        # 讀取檔案內容
        file_content = data_file.read().decode("utf-8")
        # 將檔案內容寫入臨時檔案
        data_path = f"temp_{data_file.name}"
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(file_content)
        # 驗證 JSON 格式
        json_data = json.loads(file_content)
        # 檢查結構
        if not isinstance(json_data, list) or not json_data:
            return "錯誤：JSON 檔案必須是一個非空陣列！", None, None, None, None, None
        for item in json_data:
            if not isinstance(item, dict) or "text" not in item or "label" not in item:
                return "錯誤：JSON 檔案每筆資料必須包含 'text' 和 'label' 欄位！", None, None, None, None, None
            if not isinstance(item["label"], int) or item["label"] not in [0, 1, 2, 3]:
                return "錯誤：'label' 必須是 0, 1, 2 或 3 的整數！", None, None, None, None, None
                
        # 使用唯一臨時檔案名稱避免衝突
        import uuid
        data_path = f"temp_{uuid.uuid4()}_{data_file.name}"
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(file_content)
    except json.JSONDecodeError:
        return "錯誤：上傳的檔案不是有效的 JSON 格式！", None, None, None, None, None
    except UnicodeDecodeError:
        return "錯誤：檔案編碼錯誤，請確保檔案使用 UTF-8 編碼！", None, None, None, None, None
    
    try:
        dataset = load_dataset("json", data_files=data_path)
        all_data = list(dataset["train"])
        if not all_data:
            return "錯誤：JSON 檔案無有效資料！", None, None, None, None, None
    except Exception as e:
        return f"錯誤：無法載入資料集，原因：{str(e)}", None, None, None, None, None
    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)
    
    def preprocess_function(examples):
        result = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
        result["labels"] = examples["label"]
        return result
    
    tokenized_train = list(map(preprocess_function, train_data))
    tokenized_val = list(map(preprocess_function, val_data))
    
    learning_curve_callback = LearningCurveCallback()
    
    training_args = TrainingArguments(
        output_dir=f"/results_{model_name.replace('/', '_') if model_source == '預設模型' else os.path.basename(model_path)}",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=train_batch_size,  # 使用動態參數
        per_device_eval_batch_size=eval_batch_size,    # 使用動態參數
        num_train_epochs=num_epochs,                   # 使用動態參數
        save_strategy="epoch",
        logging_dir='/logs',
        logging_steps=1,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        callbacks=[learning_curve_callback]
    )
    
    trainer.train()
    
    curve_path = learning_curve_callback.plot_learning_curves()
    eval_result, cm_path = evaluate_model(model, tokenizer, data_path)
    
    return "模型訓練完成！", eval_result, cm_path, curve_path, model, tokenizer

# 儲存模型
def save_model(model, tokenizer, save_name):
    if not save_name:
        save_name = "saved_model"
    model_save_dir = f"./{save_name}"
    base_save_dir = model_save_dir
    counter = 1
    while os.path.exists(model_save_dir):
        model_save_dir = f"{base_save_dir}_{counter}"
        counter += 1
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    return model_save_dir

# 預測功能（不變）
def predict_text(model_source, model_path, text):
    if model_source == "使用訓練後的模型":
        if st.session_state.trained_model is None or st.session_state.trained_tokenizer is None:
            return "錯誤：尚未訓練模型！"
        tokenizer = st.session_state.trained_tokenizer
        model = st.session_state.trained_model
    else:
        if not os.path.exists(model_path):
            return f"錯誤：指定的模型路徑 {model_path} 不存在！"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in encoding.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    result = f"預測結果: {LABELS[pred]} (信心度: {probs[pred].item():.2%})\n"
    result += f"低風險漂綠機率: {probs[0].item():.2%}\n"
    result += f"中風險漂綠機率: {probs[1].item():.2%}\n"
    result += f"高風險漂綠機率: {probs[2].item():.2%}\n"
    result += f"極高風險漂綠機率: {probs[3].item():.2%}"
    return result

# LIME解釋（不變）
def lime_explain(model_source, model_path, text):
    if model_source == "使用訓練後的模型":
        if st.session_state.trained_model is None or st.session_state.trained_tokenizer is None:
            return None
        tokenizer = st.session_state.trained_tokenizer
        model = st.session_state.trained_model
    else:
        if not os.path.exists(model_path):
            return None
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    class PredictWrapper:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
        
        def predict_proba(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            encodings = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            encodings = {k: v.to(self.device) for k, v in encodings.items()}
            with torch.no_grad():
                outputs = self.model(**encodings).logits
                probs = torch.nn.functional.softmax(outputs, dim=-1)
            return probs.cpu().numpy()
    
    wrapper = PredictWrapper(model, tokenizer)
    explainer = LimeTextExplainer(class_names=LABELS)
    
    tokens = tokenizer.tokenize(text)
    if len(tokens) < 2:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "錯誤：輸入文本過短，無法生成LIME解釋", ha='center', va='center', fontsize=12)
        plt.axis('off')
        lime_path = "lime_explanation.png"
        plt.savefig(lime_path)
        plt.close()
        return lime_path
    
    try:
        exp = explainer.explain_instance(text, wrapper.predict_proba, num_features=10, num_samples=50)
        plt.figure(figsize=(10, 6))
        exp.as_pyplot_figure()
        plt.title("LIME 詞彙重要性分析")
        plt.xlabel("重要性")
        plt.ylabel("詞彙")
        plt.tight_layout()
        lime_path = "lime_explanation.png"
        plt.savefig(lime_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        return lime_path
    except ValueError as e:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"錯誤：{str(e)}", ha='center', va='center', fontsize=12)
        plt.axis('off')
        lime_path = "lime_explanation.png"
        plt.savefig(lime_path)
        plt.close()
        return lime_path

# SHAP解釋（不變）
def shap_explain(model_source, model_path, text):
    if model_source == "使用訓練後的模型":
        if st.session_state.trained_model is None or st.session_state.trained_tokenizer is None:
            return None
        tokenizer = st.session_state.trained_tokenizer
        model = st.session_state.trained_model
    else:
        if not os.path.exists(model_path):
            return None
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    def get_word_importance(text):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        encoding = tokenizer(text, return_tensors="pt")
        encoding = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad():
            base_output = model(**encoding).logits
            base_probs = torch.nn.functional.softmax(base_output, dim=-1)[0].cpu()
        tokens = tokenizer.tokenize(text)
        importances = []
        for i in range(len(tokens)):
            modified_tokens = tokens.copy()
            modified_tokens[i] = tokenizer.mask_token
            modified_text = tokenizer.convert_tokens_to_string(modified_tokens)
            modified_encoding = tokenizer(modified_text, return_tensors="pt")
            modified_encoding = {k: v.to(device) for k, v in modified_encoding.items()}
            with torch.no_grad():
                modified_output = model(**modified_encoding).logits
                modified_probs = torch.nn.functional.softmax(modified_output, dim=-1)[0].cpu()
            importance = [base_probs[j] - modified_probs[j] for j in range(4)]
            importances.append((tokens[i], importance))
        return importances
    
    try:
        important_words = get_word_importance(text)
    except Exception as e:
        st.error(f"分析時發生錯誤：{str(e)}")
        return None
    
    if not important_words:
        plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, "錯誤：無法找到重要詞彙，可能是文本過短或模型無法分析", ha='center', va='center', fontsize=12)
        plt.axis('off')
        shap_path = "shap_explanation.png"
        plt.savefig(shap_path)
        plt.close()
        return shap_path
    
    generated_paths = []
    for class_idx, class_name in enumerate(LABELS):
        sorted_words = sorted(important_words, key=lambda x: abs(x[1][class_idx]), reverse=True)[:10]
        words, class_importances = zip(*[(w, imp[class_idx]) for w, imp in sorted_words])
        plt.figure(figsize=(12, 6))
        colors = ['green' if imp > 0 else 'red' for imp in class_importances]
        plt.barh(range(len(words)), [abs(imp) for imp in class_importances], color=colors)
        plt.yticks(range(len(words)), words)
        plt.xlabel('重要性值絕對值')
        plt.title(f'字詞重要性分析 - {class_name} (綠色=支持該類別, 紅色=不支持該類別)')
        plt.tight_layout()
        shap_path = f"shap_explanation_{class_name}.png"
        plt.savefig(shap_path)
        plt.close()
        generated_paths.append(shap_path)
    
    return generated_paths[0] if generated_paths else None

# Streamlit 介面
st.title("漂綠檢測模型訓練與分析")
page = st.sidebar.selectbox("選擇功能", ["模型訓練與評估", "即時預測", "LIME分析與SHAP分析"])

st.write(f"當前是否有訓練模型: {st.session_state.trained_model is not None}")
if page == "模型訓練與評估":
    st.header("模型訓練與評估")
    
    model_source = st.radio("模型來源", ["預設模型", "自定義模型路徑"])
    if model_source == "預設模型":
        model_name = st.selectbox("選擇預設模型", list(MODEL_OPTIONS.keys()))
        model_path = None
    else:
        model_name = None
        model_path = st.text_input("自定義模型路徑", placeholder="請輸入模型路徑（例如 ./my_model）")
    
    data_file = st.file_uploader("上傳數據檔案（JSON格式）", type=["json"])
    st.info("請上傳 JSON 檔案，格式應為：[{'text': '示例文字', 'label': 0}, ...]，其中 label 為 0, 1, 2 或 3，且檔案大小不超過 50MB。")
    
    # 添加訓練參數調整區域
    st.subheader("訓練參數設定")
    train_batch_size = st.slider("訓練批次大小 (per_device_train_batch_size)", min_value=1, max_value=32, value=8, step=1)
    eval_batch_size = st.slider("評估批次大小 (per_device_eval_batch_size)", min_value=1, max_value=32, value=8, step=1)
    num_epochs = st.slider("訓練週期數 (num_train_epochs)", min_value=1, max_value=10, value=3, step=1)
    
    if st.button("訓練模型"):
        with st.spinner("正在訓練模型..."):
            train_result, eval_result, cm_path, curve_path, model, tokenizer = train_model(
                model_source, model_name, model_path, data_file, train_batch_size, eval_batch_size, num_epochs
            )
        
        st.session_state.trained_model = model
        st.session_state.trained_tokenizer = tokenizer
        
        st.write(train_result)
        st.subheader("模型評估結果")
        st.write(eval_result)
        if cm_path:
            st.image(cm_path, caption="混淆矩陣")
        st.subheader("學習曲線")
        if curve_path:
            st.image(curve_path, caption="學習曲線")
    
    st.subheader("儲存模型")
    if 'save_model_clicked' not in st.session_state:
        st.session_state.save_model_clicked = False

    if not st.session_state.save_model_clicked:
        if st.session_state.trained_model is not None and st.session_state.trained_tokenizer is not None:
            if st.button("儲存模型"):
                st.session_state.save_model_clicked = True
                st.write("已點擊儲存模型，進入儲存流程")
        else:
            st.button("儲存模型", disabled=True)
            st.info("請先訓練模型以啟用儲存功能")

    if st.session_state.save_model_clicked:
        if st.session_state.trained_model is None or st.session_state.trained_tokenizer is None:
            st.warning("尚未訓練模型，無法儲存")
            st.session_state.save_model_clicked = False
        else:
            save_name = st.text_input("儲存模型名稱", placeholder="請輸入儲存模型的名稱（例如 my_trained_model）")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("確認儲存"):
                    if save_name:
                        try:
                            model_save_dir = save_model(st.session_state.trained_model, st.session_state.trained_tokenizer, save_name)
                            st.session_state.model_path = model_save_dir
                            st.success(f"模型已儲存至 {model_save_dir}")
                            st.session_state.save_model_clicked = False
                        except Exception as e:
                            st.error(f"儲存模型時發生錯誤：{str(e)}")
                    else:
                        st.warning("請輸入模型名稱")
            with col2:
                if st.button("取消"):
                    st.session_state.save_model_clicked = False
                    st.info("取消儲存模型")

elif page == "即時預測":
    st.header("即時預測")
    model_source = st.radio("模型來源", ["使用訓練後的模型", "自定義模型路徑"])
    if model_source == "自定義模型路徑":
        model_path = st.text_input("模型路徑", placeholder="請輸入已訓練模型的路徑")
    else:
        model_path = None
    text_input = st.text_area("輸入文字", height=150)
    if st.button("進行預測"):
        result = predict_text(model_source, model_path, text_input)
        st.write(result)

elif page == "LIME分析與SHAP分析":
    st.header("LIME分析與SHAP分析")
    model_source = st.radio("模型來源", ["使用訓練後的模型", "自定義模型路徑"])
    if model_source == "自定義模型路徑":
        model_path = st.text_input("模型路徑", placeholder="請輸入已訓練模型的路徑")
    else:
        model_path = None
    text_input = st.text_area("輸入文字", height=150)
    if st.button("生成LIME解釋與SHAP解釋"):
        lime_path = lime_explain(model_source, model_path, text_input)
        if lime_path:
            st.image(lime_path, caption="LIME 結果")
        shap_path = shap_explain(model_source, model_path, text_input)
        if shap_path:
            st.image(shap_path, caption="SHAP 結果")
