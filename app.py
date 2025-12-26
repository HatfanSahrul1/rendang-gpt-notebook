import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Chef Rendang AI", page_icon="🍛")
st.title("🍛 Konsultasi Resep Rendang AI")
st.write("Tanyakan apa saja soal rendang, Chef AI siap menjawab!")

# Pilih versi model
model_version = st.radio(
    "Pilih versi model",
    options=[
        "Model 1.0",
        "Model 2.0",
    ],
    index=0,
    horizontal=True,
)

ADAPTER_PATH = (
    "./gpt2-rendang-final"
    if "1.0" in model_version
    else "./gpt2-rendang-final-850"
)

# --- LOAD MODEL (Hanya sekali biar tidak berat) ---
# Fungsi ini dikasih @st.cache_resource supaya model gak di-load ulang tiap kita ngetik
@st.cache_resource
def load_model(adapter_path: str):
    BASE_MODEL_NAME = "flax-community/gpt2-small-indonesian"

    # Cek device (CPU/GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)
    model.eval()

    return model, tokenizer, device

# Loading indicator
with st.spinner(f"Sedang memanggil Chef versi {model_version} dari dapur..."):
    model, tokenizer, device = load_model(ADAPTER_PATH)
st.caption(f"Model aktif: {model_version}")

# --- INTERFACE CHAT ---
# Simpan riwayat chat di session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan chat yang sudah lalu
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input user
if prompt := st.chat_input("Tanya resep rendang..."):
    # Tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Mikir dan Jawab
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Format prompt sesuai training
        input_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        inputs = tokenizer(input_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.4,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Bersihkan hasil
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### Response:" in result:
            answer = result.split("### Response:\n")[1].strip()
        else:
            answer = result

        # Tampilkan
        message_placeholder.markdown(answer)
    
    # Simpan jawaban bot
    st.session_state.messages.append({"role": "assistant", "content": answer})