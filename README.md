# Project LLM

Participant : Lououngou Michael, Da Fonseca Alexis, Bensalah Lockman, Abdelli Khalid

## But : fine-tuner un algorithme de traitement de langage sur un modèle déjà pré-entrainé GPT2. ( Le modèle est disponible en téléchargement gpt2_lora_finetuned.zip )

Notre projet se divise en 4 parties :

- La première consiste à importer le modèle et le tester sur une toute petite génération de texte. Le but de cette partie est de constater que le modèle n'est pas correctement entraîné pour notre usage et qu'il va falloir le fine-tuner.

- La seconde consiste simplement a importer les données sur notre notebook pour pouvoir travailler avec. Un lien de téléchargement est proposé dans le notebook pour éviter de les importer directement sur Github.

- La troisième partie est la partie la plus importante du projet puisqu'elle correspond au fine-tuning de notre algorithme. En effet, nous allons pour cette partie utiliser LoRA pour le réaliser et obtenir un bon entraînement.

- La quatrième partie sera une partie évaluation du modèle et la création de l'application de notre Chatbot.
Pour lancer l'application, ouvrez un terminal dans votre éditeur de code, placez vous dans le dossier où se trouve votre fichier et exécutez la commande suivante :

```bash
streamlit run Quatrieme_partie.py

