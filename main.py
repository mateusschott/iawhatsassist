from flask import Flask, request
import pandas as pd
from twilio.twiml.messaging_response import MessagingResponse
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
from pydub import AudioSegment
import pydub

app = Flask(__name__)

# Carregar o dataset das peças
df = pd.read_csv('dataset.csv')

# Carregar os dados de treino e pré-processá-los
with open('treinamento.txt', 'r') as f:
    treinamento = [linha.strip().lower() for linha in f]

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(nltk.corpus.stopwords.words('portuguese'))

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

treinamento_preprocessed = [preprocess_text(frase) for frase in treinamento]

# Inicializar o vetorizador TF-IDF
vectorizer = TfidfVectorizer()

# Treinar o modelo de similaridade
treinamento_tfidf = vectorizer.fit_transform(treinamento_preprocessed)

# Função para gerar resposta usando PLN
def gerar_resposta(mensagem_usuario, is_treinamento=False, novos_dados_treinamento=None):
    print("Mensagem do usuário:", mensagem_usuario)  # Imprimir a mensagem original do usuário

    if is_treinamento and novos_dados_treinamento:
        with open('treinamento.txt', 'a') as f:
            for frase in novos_dados_treinamento:
                f.write(frase.strip() + '\n')  # Adiciona os novos dados como uma nova linha
        global treinamento
        treinamento.extend([preprocess_text(frase) for frase in novos_dados_treinamento])
        global treinamento_tfidf
        treinamento_tfidf = vectorizer.fit_transform(treinamento)

    mensagem_usuario_preprocessed = preprocess_text(mensagem_usuario)
    mensagem_usuario_tfidf = vectorizer.transform([mensagem_usuario_preprocessed])
    similaridades = cosine_similarity(treinamento_tfidf, mensagem_usuario_tfidf)

    print("Similaridades:", similaridades)

    indice_mais_similar = similaridades.argmax()
    resposta_similar = treinamento[indice_mais_similar]

    print("Resposta similar:", resposta_similar)

    return resposta_similar

@app.route('/bot', methods=['POST'])
def bot():
    resp = MessagingResponse()
    msg = resp.message()
    responded = False

    incoming_msg = request.values.get('Body', '').lower()  # Definir incoming_msg aqui

    if 'MediaUrl0' in request.values:
        print("Áudio recebido!")
        import requests
        audio_url = request.values.get('MediaUrl0')
        account_sid = ''
        auth_token = ''

        response = requests.get(audio_url, auth=(account_sid, auth_token))

        if response.status_code == 200:
            with open('audio.wav', 'wb') as f:
                f.write(response.content)
            print("Áudio baixado com sucesso!")

            # Converter o arquivo de áudio para WAV usando pydub
            audio_file = pydub.AudioSegment.from_file('audio.wav')
            audio_file.export('audio.wav', format='wav')

            # Processar o arquivo de áudio com SpeechRecognition
            r = sr.Recognizer()
            with sr.AudioFile('audio.wav') as source:
                audio = r.record(source)
                try:
                    incoming_msg = r.recognize_google(audio, language='pt-BR')
                    print("Mensagem do usuário:", incoming_msg)
                except sr.UnknownValueError:
                    print("Não foi possível entender o áudio")
                except sr.RequestError as e:
                    print("Erro ao reconhecer o áudio:", e)

            # Processar a mensagem do usuário como texto
            responded = False

    # Introdução e saudação
    if "oi" in incoming_msg or "olá" in incoming_msg:
        resposta = "Olá, informe o ID ou o nome do produto!"
        msg.body(resposta)
        responded = True

    # Buscar informações sobre a peça solicitada
    if not responded:
        part = None
        if incoming_msg.isdigit():
            part_id = int(incoming_msg)
            part = df[df['part_id'] == part_id]
            part_name = df.loc[part.index, 'part_name'].values[0]  # Define part_name aqui
        else:
            part_name = incoming_msg.title()
            part = df[df['part_name'] == part_name]

        if not part.empty:
            part_info = f"ID: {part['part_id'].values[0]}\nNome: {part_name}\nQuantidade: {part['quantity'].values[0]}\nPreço: {part['price'].values[0]}"
            msg.body(part_info)
            responded = True

    # Verificar se é uma mensagem de treinamento
    is_treinamento = False
    if not responded:
        if "treinar" in incoming_msg:
            is_treinamento = True
            responded = True

    # Se é uma mensagem de treinamento, extrair os novos dados de treinamento
    if is_treinamento: #and not responded:
        novos_dados_treinamento = [frase.strip() for frase in incoming_msg.split("treinar", 1)[1].split("\n")]
        # Chamamos a função gerar_resposta passando is_treinamento como True e os novos dados de treinamento
        resposta = gerar_resposta(incoming_msg, is_treinamento=True, novos_dados_treinamento=novos_dados_treinamento)
        msg.body("Treinamento concluído com sucesso.")


    # Gerar uma resposta caso nenhuma das condições anteriores tenha sido satisfeita
    if not responded:
        resposta = gerar_resposta(incoming_msg)
        msg.body(resposta)
        responded = True

    return str(resp)

if __name__ == '__main__':
    app.run()