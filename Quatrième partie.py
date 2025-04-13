#Streamlit permet de créer des applications web interactives en Python, et est particulièrement adapté pour les projets de science des données et d'apprentissage automatique.
# Il est utilisé ici pour créer une interface utilisateur simple pour interagir avec le modèle de langage.
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
# Chargement dU modèle de base GPT2
tokenizer = AutoTokenizer.from_pretrained("gpt2")
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
base_model.eval() 
# Chargement de l'adaptateur LoRA
adapter_path = "C:/chemin/acces/modele/lora"  # Remplacer par le chemin vers l'adaptateur LoRA

# Chargement du modèle LoRA
config = PeftConfig.from_pretrained(adapter_path)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval() 

# Configuration de la page web Streamlit
st.set_page_config(page_title="Chat IA", page_icon="🤖")

# Titre principal affiché sur la page
st.title("🤖 Chat avec ton modèle IA")

with st.form(key="chat_form"):
    # Zone de texte pour entrer une question ou une consigne pour l'IA
    prompt_input = st.text_area(
        "Pose une question à l'IA :",  # label affiché
        height=150,                    # hauteur de la zone de texte
        value="Tell me 3 things to do in Istanbul ?"  # texte pré-rempli
    )
    
    # Bouton pour envoyer la requête
    submitted = st.form_submit_button("Envoyer")

if submitted:
    # Ajoute le format d'instruction du prompt
    full_prompt = f"### Instruction:\n{prompt_input}\n\n### Response:\n"
    
    # Tokenisation
    inputs = tokenizer(full_prompt, return_tensors="pt")
    ## Générer une réponse avec le modèle LoRA et GPT2, on prend les 50 meilleurs mots de la distribution avec une température de 0.8
    # et une probabilité cumulée(top_p) d'au moins 0.95
    # et une longueur maximale de 100 tokens
    with torch.no_grad():
        outputs = model.generate(**inputs,
                                 do_sample=True,
                                 temperature=0.8,
                                 max_length=100,
                                 top_p=0.95,
                                 top_k=50,
                                 num_return_sequences=1)

        outputs2 = base_model.generate(**inputs,
                                       do_sample=True,
                                       temperature=0.8,
                                       max_length=100,
                                       top_p=0.95,
                                       top_k=50,
                                       num_return_sequences=1)

    # Décodage de la réponse du modèle LoRA
    response_lora = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Décodage de la réponse du modèle de base
    response_base = tokenizer.decode(outputs2[0], skip_special_tokens=True)

    # Affichage stylisé avec Markdown pour introduire la section LoRA
    st.markdown("### Réponse du modèle LoRA :")
    
    # Encadré vert pour mettre en valeur la réponse principale (succès)
    st.success(response_lora)

    # Affichage Markdown pour la comparaison
    st.markdown("### Réponse du modèle de base (GPT2) :")

    # Encadré bleu pour une information secondaire (réponse du modèle de base)
    st.info(response_base)
